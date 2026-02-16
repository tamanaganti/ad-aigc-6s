#!/usr/bin/env python3
"""
Lambda sensitivity runner aligned to your main.py CLI.

Runs:
  for lambda in datasets_dir/*.csv:
    for fusion in [concat, standard_attn, quality_gated]:
      for seed in [42, 123, 999]:
        python main.py --config ... --dataset_csv ... --fusion_type ... --seed ... --output_dir ... --skip_train_if_exists

Collects:
  MAE from {output_dir}/test_predictions.csv (p_retained vs p_pred)

Outputs:
  experiments/lambda_sensitivity/lambda_sensitivity_per_run.csv
  experiments/lambda_sensitivity/lambda_sensitivity_summary.csv
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS = [42, 123, 999]
FUSIONS = {
    "Concatenation + MLP": "concat",
    "Standard Attention": "standard_attn",
    "Quality-Gated Attention": "quality_gated",
}


def parse_lambda_from_name(p: Path) -> float:
    # expects: ad_aigc_6s_main_experiment_lambda_0.50.csv
    key = "_lambda_"
    if key not in p.stem:
        raise ValueError(f"Could not parse lambda from filename: {p.name}")
    return float(p.stem.split(key, 1)[1])


def mae_from_preds(pred_csv: Path) -> float:
    df = pd.read_csv(pred_csv)
    if "p_retained" not in df.columns or "p_pred" not in df.columns:
        raise ValueError(f"{pred_csv} missing required columns. Found: {list(df.columns)}")
    return float(np.mean(np.abs(df["p_retained"].to_numpy() - df["p_pred"].to_numpy())))


def run_one(repo_root: Path, config_path: Path, dataset_csv: Path, fusion: str, seed: int, out_dir: Path, num_workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--config", str(config_path),
        "--dataset_csv", str(dataset_csv),
        "--fusion_type", fusion,
        "--seed", str(seed),
        "--output_dir", str(out_dir),
        "--num-workers", str(num_workers),
        "--skip_train_if_exists",
    ]

    res = subprocess.run(cmd, env=env, cwd=str(repo_root))
    if res.returncode != 0:
        raise RuntimeError(f"FAILED (main.py exit code {res.returncode})")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--datasets_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    datasets_dir = (repo_root / args.datasets_dir).resolve()
    out_root = (repo_root / args.out_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(datasets_dir.glob("*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(f"No CSVs found in: {datasets_dir}")

    rows = []
    for ds in dataset_paths:
        lam = parse_lambda_from_name(ds)

        for model_name, fusion in FUSIONS.items():
            for seed in SEEDS:
                run_dir = out_root / f"lambda_{lam:.2f}" / f"{fusion}_{seed}"
                pred_csv = run_dir / "test_predictions.csv"

                try:
                    run_one(repo_root, config_path, ds, fusion, seed, run_dir, num_workers=args.num_workers)
                    m = mae_from_preds(pred_csv)
                    rows.append({
                        "lambda": lam,
                        "model": model_name,
                        "fusion": fusion,
                        "seed": seed,
                        "mae": m,
                        "pred_csv": str(pred_csv),
                    })
                    print(f"lambda={lam:.2f} | {fusion} | seed={seed} | MAE={m:.6f}")
                except Exception as e:
                    rows.append({
                        "lambda": lam,
                        "model": model_name,
                        "fusion": fusion,
                        "seed": seed,
                        "mae": np.nan,
                        "pred_csv": "",
                        "error": str(e),
                    })
                    print(f"lambda={lam:.2f} | {fusion} | seed={seed} | FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "lambda_sensitivity_per_run.csv", index=False)

    # Aggregate mean/std per (lambda, model)
    agg = (
        df.dropna(subset=["mae"])
          .groupby(["lambda", "model"], as_index=False)
          .agg(
              mae_mean=("mae", "mean"),
              mae_std=("mae", lambda x: float(np.std(x.to_numpy(), ddof=0))),
              n=("mae", "count"),
          )
          .sort_values(["lambda", "model"])
    )
    agg.to_csv(out_root / "lambda_sensitivity_summary.csv", index=False)

    print(f"Wrote: {out_root / 'lambda_sensitivity_per_run.csv'}")
    print(f"Wrote: {out_root / 'lambda_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
