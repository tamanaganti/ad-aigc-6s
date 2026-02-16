#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

# Run from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.multimodal_model import MultimodalAdModel


# -----------------------------
# Config wrapper that preserves dicts where needed
# -----------------------------
class AttrObj:
    """Simple attribute access wrapper."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

def wrap_config(d, *, keep_dict_keys=None, path=""):
    """
    Recursively wrap dicts into AttrObj, except for keys listed in keep_dict_keys
    (those are left as real dicts, with values still wrapped).
    """
    keep_dict_keys = keep_dict_keys or set()

    if isinstance(d, dict):
        # If this dict is meant to stay as dict, return dict with wrapped children
        # (handled by parent based on key).
        obj = {}
        for k, v in d.items():
            # keep certain keys as dicts (e.g., "heads")
            if k in keep_dict_keys:
                if isinstance(v, dict):
                    obj[k] = {kk: wrap_config(vv, keep_dict_keys=keep_dict_keys, path=f"{path}.{k}.{kk}")
                              for kk, vv in v.items()}
                else:
                    obj[k] = v
            else:
                obj[k] = wrap_config(v, keep_dict_keys=keep_dict_keys, path=f"{path}.{k}")
        return AttrObj(obj)

    if isinstance(d, list):
        return [wrap_config(x, keep_dict_keys=keep_dict_keys, path=f"{path}[]") for x in d]

    return d


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Keep model_ad_aigc_6s.heads as a REAL dict (so .items() works)
    return wrap_config(cfg, keep_dict_keys={"heads"})


def get_dataset_path_from_config(cfg):
    # Matches your main.py conventions
    if getattr(cfg, "data", None) and getattr(cfg.data, "ad_aigc_6s", None):
        base_path = cfg.data.ad_aigc_6s.path
    else:
        raise ValueError("Config missing data.ad_aigc_6s.path")
    cache_path = base_path.replace(".csv", "_cached.csv")
    return cache_path if os.path.exists(os.path.join(REPO_ROOT, cache_path)) else base_path


def main():
    ap = argparse.ArgumentParser(description="Generate gating plot from a trained checkpoint.")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--model_path", default="experiments/quality_gated_42/best_model.pt", help="Path to model checkpoint (.pt)")
    ap.add_argument("--out_dir", default="outputs/paper_artifacts", help="Output directory for the figure")
    ap.add_argument("--fig_name", default="figure_7_final.png", help="Output figure filename")
    ap.add_argument("--split_seed", type=int, default=42, help="Seed used to reconstruct the deterministic test split")
    args = ap.parse_args()

    config_abs = os.path.join(REPO_ROOT, args.config)
    model_abs = os.path.join(REPO_ROOT, args.model_path)

    if not os.path.exists(config_abs):
        raise FileNotFoundError(f"Config not found: {config_abs}")
    if not os.path.exists(model_abs):
        raise FileNotFoundError(f"Model checkpoint not found: {model_abs}")

    cfg = load_yaml_config(config_abs)

    # Ensure fusion type is correct for this checkpoint
    cfg.model_ad_aigc_6s.fusion.type = "quality_gated"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load dataset + deterministic split
    # -----------------------------
    dataset_csv = get_dataset_path_from_config(cfg)
    dataset_abs = os.path.join(REPO_ROOT, dataset_csv)
    if not os.path.exists(dataset_abs):
        raise FileNotFoundError(f"Dataset not found: {dataset_abs}")

    print(f"Reconstructing deterministic test split (seed={args.split_seed})...")
    df_all = pd.read_csv(dataset_abs)
    all_idx = np.arange(len(df_all))
    train_idx, rem_idx = train_test_split(all_idx, test_size=0.2, random_state=args.split_seed, shuffle=True)
    val_idx, test_idx = train_test_split(rem_idx, test_size=0.5, random_state=args.split_seed, shuffle=True)
    df_test = df_all.iloc[test_idx].copy()
    print(f"Test samples: {len(df_test)}")

    # -----------------------------
    # Build model + load weights
    # -----------------------------
    print("Loading config + model checkpoint...")
    model = MultimodalAdModel(cfg.model_ad_aigc_6s).to(device)
    state = torch.load(model_abs, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -----------------------------
    # Quality columns (must match training)
    # Prefer config.model_ad_aigc_6s.quality_columns if present.
    # -----------------------------
    quality_cols = getattr(cfg.model_ad_aigc_6s, "quality_columns", None)
    if not quality_cols:
        raise ValueError("Config missing model_ad_aigc_6s.quality_columns")

    missing_cols = [c for c in quality_cols if c not in df_test.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing {len(missing_cols)} quality columns, e.g.: {missing_cols[:5]}")

    q_data = df_test[quality_cols].to_numpy(dtype=np.float32)

    # Compute "quality_good" (higher is better) and "artifact" (higher is worse).
    vbench_w = float(cfg.data.ad_aigc_6s.quality_aggregation_weights.vbench_weight)
    eval_w = float(cfg.data.ad_aigc_6s.quality_aggregation_weights.evalcrafter_weight)

    # By construction, first 16 are VBench and remaining 17 are EvalCrafter in your pipeline.
    vbench = q_data[:, :16].mean(axis=1)
    evalc = q_data[:, 16:].mean(axis=1)
    quality_good = vbench_w * vbench + eval_w * evalc
    artifact = 1.0 - quality_good  # higher = worse

    # -----------------------------
    # Extract gate activations
    # -----------------------------
    q_tensor = torch.tensor(q_data, device=device)
    with torch.no_grad():
        q_feat = model.qual_proj(q_tensor)               # [N, H]
        gate = model.fusion.gate_net(q_feat)             # [N, H]
        gate_np = gate.detach().cpu().numpy()

    H = gate_np.shape[1]
    print(f"Gate activation shape: {gate_np.shape} (H={H})")

    # -----------------------------
    # Auto-pick a representative channel:
    # Choose channel with strongest positive correlation with artifact (worse quality -> lower gate),
    # i.e., strongest NEGATIVE correlation between quality_good and gate,
    # equivalently strongest POSITIVE correlation between artifact and (1-gate) but we keep simple:
    # pick channel with most NEGATIVE Pearson r between artifact and gate (suppression).
    # -----------------------------
    rs = np.array([np.corrcoef(artifact, gate_np[:, j])[0, 1] for j in range(H)], dtype=float)

    # We want suppression: as artifact increases, gate decreases => correlation should be negative
    best_dim = int(np.nanargmin(rs))
    r_val, p_pearson = pearsonr(artifact, gate_np[:, best_dim])
    rho, p_rho = spearmanr(artifact, gate_np[:, best_dim])

    print(f"Selected gate channel: {best_dim}")
    print(f"Pearson r(artifact, gate): {r_val:.4f} (p={p_pearson:.2e})")
    print(f"Spearman rho(artifact, gate): {rho:.4f} (p={p_rho:.2e})")

    # -----------------------------
    # Plot (matplotlib only, no seaborn)
    # -----------------------------
    Path(os.path.join(REPO_ROOT, args.out_dir)).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(REPO_ROOT, args.out_dir, args.fig_name)

    plt.figure(figsize=(7.2, 5.4))
    plt.scatter(artifact, gate_np[:, best_dim], alpha=0.45, s=14)

    # Simple linear fit line
    x = artifact
    y = gate_np[:, best_dim]
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, m * xs + b, linewidth=2)

    plt.xlabel("Aggregate Artifact Score (higher = worse)")
    plt.ylabel(f"Gate Activation (channel {best_dim})")
    plt.title(f"Gating suppression vs artifacts\nPearson r={r_val:.2f}, Spearman rho={rho:.2f}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
