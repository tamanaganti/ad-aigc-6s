# main.py
#!/usr/bin/env python3
"""
Main Entry Point for Ad-AIGC-6s Training Pipeline.

Drop-in replacement goals:
- No dependency on src.config (ConfigStruct is defined here)
- Allow CLI overrides so robustness runs do NOT edit config.yaml
- Make src imports work regardless of where main.py is launched from
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ensure repo root is in python path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.reproducibility import set_deterministic
from src.data.data_processor import DataProcessor

try:
    from src.training.trainer import Trainer
except ImportError:
    from src.training.training import Trainer  # fallback if repo uses an alternate module name

from src.models.multimodal_model import MultimodalAdModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ConfigStruct:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                self.__dict__[key] = ConfigStruct(**value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

    def __getitem__(self, name):
        return self.__dict__.get(name, None)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


def load_config(path: str) -> ConfigStruct:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return ConfigStruct(**cfg_dict)


def _get_device(config: ConfigStruct) -> torch.device:
    device_str = getattr(config, "device", None)
    if torch.cuda.is_available():
        device_str = device_str or "cuda"
    else:
        device_str = "cpu"
    return torch.device(device_str)


def get_test_ids_from_csv(config: ConfigStruct) -> list:
    """
    Re-implements the split logic to recover the exact Test IDs.
    This bypasses any dataset ordering ambiguity.
    """
    # Locate dataset config
    if getattr(config, "data", None) and getattr(config.data, "ad_aigc_6s", None):
        data_cfg = config.data.ad_aigc_6s
    elif getattr(config, "ad_aigc_6s", None):
        data_cfg = config.ad_aigc_6s
    else:
        raise ValueError("Config missing data.ad_aigc_6s (or ad_aigc_6s)")

    base_path = data_cfg.path
    cache_path = base_path.replace(".csv", "_cached.csv")
    load_path = cache_path if os.path.exists(cache_path) else base_path

    logger.info(f"Reloading Test IDs from: {load_path}")
    df = pd.read_csv(load_path)

    train_ratio = float(getattr(data_cfg, "train_split", 0.8) or 0.8)
    val_ratio = float(getattr(data_cfg, "val_split", 0.1) or 0.1)
    test_ratio = float(getattr(data_cfg, "test_split", 1.0 - train_ratio - val_ratio) or (1.0 - train_ratio - val_ratio))

    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    train_df, rest_df = train_test_split(df, test_size=(1.0 - train_ratio), random_state=42, shuffle=True)
    val_frac_of_rest = val_ratio / (val_ratio + test_ratio + 1e-12)
    val_df, test_df = train_test_split(rest_df, test_size=(1.0 - val_frac_of_rest), random_state=42, shuffle=True)

    if "ad_id" not in test_df.columns:
        raise ValueError(f"Dataset CSV missing ad_id. Columns: {list(test_df.columns)}")

    ids = test_df["ad_id"].astype(str).tolist()
    logger.info(f"Recovered {len(ids)} Test IDs.")
    return ids


def generate_test_artifacts(model, test_dataset, config: ConfigStruct, seed: int = 42, num_workers: int = 0) -> None:
    device = _get_device(config)
    model.to(device)
    model.eval()

    test_ids = get_test_ids_from_csv(config)

    gen = torch.Generator()
    gen.manual_seed(seed)

    loader = DataLoader(
        test_dataset,
        batch_size=int(getattr(config.training, "batch_size", 128) or 128),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=gen if int(num_workers) > 0 else None,
    )

    if len(test_ids) != len(test_dataset):
        logger.warning(f"ID mismatch: CSV has {len(test_ids)}, Dataset has {len(test_dataset)}. Falling back to dummy IDs.")
        test_ids = [f"unknown_{i}" for i in range(len(test_dataset))]

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model(batch_gpu)

            if "ad_aigc_6s_ai" in outputs:
                pred = outputs["ad_aigc_6s_ai"]
            else:
                pred = list(outputs.values())[0]

            if pred.dim() > 1:
                pred = pred.squeeze(-1)

            all_preds.extend(pred.detach().float().cpu().numpy().tolist())
            all_targets.extend(batch_gpu["p_retained"].detach().float().cpu().numpy().tolist())

    df_results = pd.DataFrame(
        {
            "ad_id": test_ids[: len(all_preds)],
            "p_retained": all_targets,
            "p_pred": all_preds,
        }
    )

    output_dir = config.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "test_predictions.csv")
    df_results.to_csv(pred_path, index=False)

    logger.info(f"Saved Test Predictions: {pred_path} (rows={len(df_results)})")


def _apply_cli_overrides(config: ConfigStruct, args: argparse.Namespace) -> None:
    # dataset path override
    if args.dataset_csv:
        if not (getattr(config, "data", None) and getattr(config.data, "ad_aigc_6s", None)):
            raise ValueError("config.data.ad_aigc_6s missing, cannot override dataset path")
        config.data.ad_aigc_6s.path = args.dataset_csv

    # fusion override
    if args.fusion_type:
        if not (getattr(config, "model_ad_aigc_6s", None) and getattr(config.model_ad_aigc_6s, "fusion", None)):
            raise ValueError("config.model_ad_aigc_6s.fusion missing, cannot override fusion type")
        config.model_ad_aigc_6s.fusion.type = args.fusion_type

    # seed override
    if args.seed is not None:
        config.seed = int(args.seed)

    # output dir override
    if args.output_dir:
        if not getattr(config, "training", None):
            config.training = ConfigStruct()
        config.training.output_dir = args.output_dir

    # device override (optional)
    if args.device:
        config.device = args.device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num-workers", type=int, default=0)

    # Overrides so robustness runs do not edit config.yaml
    parser.add_argument("--fusion_type", type=str, default=None, choices=["concat", "standard_attn", "quality_gated"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_csv", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Behavior flags
    parser.add_argument("--skip_train_if_exists", action="store_true", help="If best_model.pt exists, reuse it.")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    _apply_cli_overrides(config, args)

    seed = int(getattr(config, "seed", 42) or 42)
    set_deterministic(seed)

    if not getattr(config, "training", None):
        raise ValueError("config.training missing")
    if not getattr(config.training, "output_dir", None):
        raise ValueError("config.training.output_dir missing (or not overridden)")

    os.makedirs(config.training.output_dir, exist_ok=True)

    logger.info("Initializing Data Pipeline...")
    processor = DataProcessor(config)
    datasets = processor.load_dataset_from_csv()

    logger.info("Initializing Model...")
    model = MultimodalAdModel(config.model_ad_aigc_6s)

    trainer = Trainer(config)
    best_model_path = os.path.join(config.training.output_dir, "best_model.pt")

    if args.skip_train_if_exists and os.path.exists(best_model_path):
        logger.info(f"Found existing best model at {best_model_path}, loading...")
        model = trainer.load_model(model, best_model_path)
        best_model = model
    else:
        logger.info("Starting Training Loop...")
        best_model = trainer.train(model, datasets)

    logger.info("Generating Final Test Predictions...")
    generate_test_artifacts(best_model, datasets["test"], config, seed=seed, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
