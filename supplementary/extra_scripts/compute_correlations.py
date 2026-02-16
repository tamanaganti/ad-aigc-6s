# scripts/compute_correlations.py
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--target", default="retention_factor", choices=["p_retained", "retention_factor", "q_ai"])
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv)

    if args.target == "p_retained":
        y = df["p_retained"].astype(float)
    elif args.target == "retention_factor":
        if "retention_factor" in df.columns:
            y = df["retention_factor"].astype(float)
        else:
            lam = float(df["lambda"].iloc[0]) if "lambda" in df.columns else 0.5
            y = lam * df["q_ai_aggregate"].astype(float) + (1.0 - lam) * df["s_sim"].astype(float)
    else:
        y = df["q_ai_aggregate"].astype(float)

    rows = []
    for m in args.metrics:
        if m not in df.columns:
            rows.append({"metric": m, "pearson_r": np.nan, "spearman_rho": np.nan})
            continue
        x = df[m].astype(float)
        rows.append({
            "metric": m,
            "pearson_r": float(x.corr(y, method="pearson")),
            "spearman_rho": float(x.corr(y, method="spearman")),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote correlations: {args.out_csv}")

if __name__ == "__main__":
    main()
