#!/usr/bin/env python3
"""
Paired bootstrap significance on MAE delta across multiple seeds.

We compare Proposed vs Baseline, using paired errors per example.
We pool rows across the provided prediction files (e.g., 3 seeds), but preserve pairing
because we always sample indices from the pooled aligned arrays.

Outputs:
- observed delta MAE (proposed - baseline)
- 95% bootstrap CI of delta
- one-sided p-value: P(delta >= 0)  (i.e., proposed not better)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept p_retained/p_pred or y_true/y_pred
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["ad_id", "id", "video_id"]:
            col_map[c] = "ad_id"
        if lc in ["p_retained", "y_true", "true", "target"]:
            col_map[c] = "y_true"
        if lc in ["p_pred", "y_pred", "pred", "prediction"]:
            col_map[c] = "y_pred"
    df = df.rename(columns=col_map)

    for col in ["ad_id", "y_true", "y_pred"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing {col}. Found: {list(df.columns)}")

    df = df[["ad_id", "y_true", "y_pred"]].copy()
    df["ad_id"] = df["ad_id"].astype(str)
    df["y_true"] = df["y_true"].astype(float)
    df["y_pred"] = df["y_pred"].astype(float)
    df = df.sort_values("ad_id").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", nargs="+", required=True, help="Proposed prediction CSVs (e.g., 3 seeds).")
    ap.add_argument("--baseline", nargs="+", required=True, help="Baseline prediction CSVs (same seeds/order).")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.proposed) != len(args.baseline):
        raise ValueError("proposed and baseline must have the same number of files (e.g., 3 seeds each).")

    y_true_all = []
    prop_all = []
    base_all = []

    for p_path, b_path in zip(args.proposed, args.baseline):
        df_p = load_pred(Path(p_path))
        df_b = load_pred(Path(b_path))

        if not np.array_equal(df_p["ad_id"].values, df_b["ad_id"].values):
            raise ValueError(f"ad_id mismatch between {p_path} and {b_path}")

        y_true_all.append(df_p["y_true"].to_numpy())
        prop_all.append(df_p["y_pred"].to_numpy())
        base_all.append(df_b["y_pred"].to_numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    pred_p = np.concatenate(prop_all, axis=0)
    pred_b = np.concatenate(base_all, axis=0)

    err_p = np.abs(y_true - pred_p)
    err_b = np.abs(y_true - pred_b)

    observed = float(np.mean(err_p) - np.mean(err_b))

    rng = np.random.default_rng(int(args.seed))
    n = len(y_true)

    diffs = np.empty(int(args.n_boot), dtype=float)
    for i in range(int(args.n_boot)):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(err_p[idx]) - np.mean(err_b[idx]))

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # one-sided: probability proposed is not better (delta >= 0)
    p_value = float(np.mean(diffs >= 0.0))

    print("STATISTICAL SIGNIFICANCE (paired bootstrap)")
    print(f"Observed delta (proposed - baseline): {observed:.6f}")
    print(f"95% CI of delta: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"One-sided p-value (P(delta >= 0)): {p_value:.6f}")


if __name__ == "__main__":
    main()
