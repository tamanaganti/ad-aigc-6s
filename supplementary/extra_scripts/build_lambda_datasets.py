#!/usr/bin/env python3
"""
Build lambda-sensitivity datasets by recomputing:
  retention_factor = lam*q_ai_aggregate + (1-lam)*s_sim
  p_retained       = p_real * retention_factor

Input CSV must include: ad_id, p_real, q_ai_aggregate (or q_ai), s_sim
Output: one CSV per lambda with updated columns: lambda, retention_factor, p_retained

Example:
  python scripts/build_lambda_datasets.py \
    --in_csv data/metadata/ad_aigc_6s_main_experiment.csv \
    --out_dir data/metadata/lambda_sensitivity \
    --lambdas 0 0.1 0.25 0.5 0.75 0.9 1.0
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # allow either q_ai_aggregate or q_ai
    if "q_ai_aggregate" in df.columns:
        q_col = "q_ai_aggregate"
    elif "q_ai" in df.columns:
        q_col = "q_ai"
    else:
        raise ValueError(f"Input CSV must include q_ai_aggregate or q_ai. Found: {list(df.columns)}")

    required = {"ad_id", "p_real", q_col, "s_sim"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}. Found: {list(df.columns)}")

    # numeric cleanup + fail fast
    for c in ["p_real", q_col, "s_sim"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["p_real", q_col, "s_sim"]].isna().any().any():
        bad = int(df[["p_real", q_col, "s_sim"]].isna().sum().sum())
        raise ValueError(f"Found NaNs in required numeric columns (count={bad}). Fix upstream CSV first.")

    for lam_s in args.lambdas:
        lam = float(lam_s)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"Lambda must be in [0,1]. Got {lam}")

        out = df.copy()
        out["lambda"] = float(lam)
        out["retention_factor"] = lam * out[q_col].astype(float) + (1.0 - lam) * out["s_sim"].astype(float)
        out["p_retained"] = out["p_real"].astype(float) * out["retention_factor"].astype(float)

        out_path = out_dir / f"ad_aigc_6s_main_experiment_lambda_{lam:.2f}.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} (rows={len(out)})")

    print("Done.")


if __name__ == "__main__":
    main()
