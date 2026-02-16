# scripts/run_robustness_check.py
#!/usr/bin/env python3
"""
Robustness check (3 models x 3 seeds), without editing config.yaml.

It runs:
- concat
- standard_attn
- quality_gated
over seeds [42, 123, 999]

Outputs:
- experiments/robustness/{fusion}_{seed}/test_predictions.csv
- experiments/robustness/robustness_summary.csv
- prints mean ± std MAE per model

Important:
- It sets PYTHONPATH to repo root so src.* imports work.
- It calls main.py with CLI overrides: --fusion_type --seed --output_dir --dataset_csv
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

SEEDS = [42, 123, 999]
FUSIONS = {
    "Concat": "concat",
    "Standard Attn": "standard_attn",
    "Quality Gated": "quality_gated",
}

DEFAULT_CONFIG = "config.yaml"
DEFAULT_DATASET = "data/metadata/ad_aigc_6s_main_experiment.csv"
OUT_ROOT = "experiments/robustness"


def mae_from_preds(pred_csv: str) -> float:
    df = pd.read_csv(pred_csv)
    if "p_retained" not in df.columns or "p_pred" not in df.columns:
        raise ValueError(f"{pred_csv} missing required columns. Found: {list(df.columns)}")
    return float(np.mean(np.abs(df["p_retained"].to_numpy() - df["p_pred"].to_numpy())))


def run_one(config_path: str, dataset_csv: str, fusion: str, seed: int, out_dir: str, num_workers: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        os.path.join(repo_root, "main.py"),
        "--config",
        os.path.abspath(os.path.join(repo_root, config_path)),
        "--dataset_csv",
        os.path.abspath(os.path.join(repo_root, dataset_csv)),
        "--fusion_type",
        fusion,
        "--seed",
        str(seed),
        "--output_dir",
        os.path.abspath(os.path.join(repo_root, out_dir)),
        "--skip_train_if_exists",
        "--num-workers",
        str(num_workers),
    ]

    print(f"Run: {fusion} | seed={seed} -> {out_dir}")
    res = subprocess.run(cmd, env=env, cwd=repo_root)
    if res.returncode != 0:
        raise RuntimeError(f"FAILED (main.py exit code {res.returncode})")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--dataset_csv", default=DEFAULT_DATASET)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    rows = []
    for model_name, fusion in FUSIONS.items():
        for seed in SEEDS:
            out_dir = os.path.join(OUT_ROOT, f"{fusion}_{seed}")
            try:
                run_one(args.config, args.dataset_csv, fusion, seed, out_dir, num_workers=args.num_workers)
                pred_csv = os.path.join(out_dir, "test_predictions.csv")
                m = mae_from_preds(pred_csv)
                rows.append({"model": model_name, "fusion": fusion, "seed": seed, "mae": m, "pred_csv": pred_csv})
                print(f"  MAE: {m:.6f}")
            except Exception as e:
                rows.append({"model": model_name, "fusion": fusion, "seed": seed, "mae": np.nan, "pred_csv": ""})
                print(f"  FAILED: {e}")

    df = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_ROOT, "robustness_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nWrote: {summary_path}")

    print("\nROBUSTNESS REPORT (MAE)")
    for model_name in FUSIONS.keys():
        vals = df[(df["model"] == model_name) & df["mae"].notna()]["mae"].to_numpy()
        if len(vals) == 0:
            print(f"{model_name}: N/A")
        else:
            print(f"{model_name}: {vals.mean():.6f} ± {vals.std(ddof=0):.6f}")


if __name__ == "__main__":
    main()
