#!/usr/bin/env python3
"""
Build missing baselines for Table I (deterministic, no training loops).

Baselines created (test set only, using the same 80/10/10 split logic):
- mean_pred: predicts mean(y_train)
- q_only_lr: LinearRegression on q_ai_aggregate
- s_only_lr: LinearRegression on s_sim
- q_s_lr:    LinearRegression on [q_ai_aggregate, s_sim]

Outputs:
experiments/baselines/<baseline_name>/test_predictions.csv with columns:
ad_id, y_true, y_pred

Note: This script uses a fixed split (random_state=42) for reproducibility.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def split_ids(df: pd.DataFrame, seed: int = 42, train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1):
    if abs(train_split + val_split + test_split - 1.0) > 1e-9:
        raise ValueError("Splits must sum to 1.0")

    # Two-stage split: 80/20 then split 20 into 10/10 (val/test)
    train_df, temp_df = train_test_split(df, test_size=(1.0 - train_split), random_state=seed, shuffle=True)
    # temp is 20%; split equally into val/test
    val_size = val_split / (val_split + test_split)  # = 0.5 when 10/10
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_size), random_state=seed, shuffle=True)

    return train_df, val_df, test_df


def write_pred(out_dir: Path, ad_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ad_id": ad_id.astype(str), "y_true": y_true.astype(float), "y_pred": y_pred.astype(float)})
    df.to_csv(out_dir / "test_predictions.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True, help="Main labeled dataset CSV (must include ad_id, p_retained, q_ai_aggregate, s_sim).")
    ap.add_argument("--out_root", default="experiments/baselines", help="Where to write baseline prediction folders.")
    ap.add_argument("--seed", type=int, default=42, help="Split seed for reproducibility.")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset_csv)
    for col in ["ad_id", "p_retained", "q_ai_aggregate", "s_sim"]:
        if col not in df.columns:
            raise ValueError(f"dataset_csv missing required column: {col}. Found: {list(df.columns)}")

    df = df.copy()
    df["ad_id"] = df["ad_id"].astype(str)
    df["p_retained"] = df["p_retained"].astype(float)
    df["q_ai_aggregate"] = df["q_ai_aggregate"].astype(float)
    df["s_sim"] = df["s_sim"].astype(float)

    train_df, _, test_df = split_ids(df, seed=args.seed)

    y_train = train_df["p_retained"].to_numpy()
    y_test = test_df["p_retained"].to_numpy()
    ad_test = test_df["ad_id"].to_numpy()

    out_root = Path(args.out_root)

    # 1) Mean prediction
    mean_y = float(np.mean(y_train))
    write_pred(out_root / "mean_pred", ad_test, y_test, np.full_like(y_test, mean_y, dtype=float))

    # 2) q-only linear regression
    lr_q = LinearRegression()
    lr_q.fit(train_df[["q_ai_aggregate"]].to_numpy(), y_train)
    pred_q = lr_q.predict(test_df[["q_ai_aggregate"]].to_numpy())
    write_pred(out_root / "q_only_lr", ad_test, y_test, pred_q)

    # 3) s-only linear regression
    lr_s = LinearRegression()
    lr_s.fit(train_df[["s_sim"]].to_numpy(), y_train)
    pred_s = lr_s.predict(test_df[["s_sim"]].to_numpy())
    write_pred(out_root / "s_only_lr", ad_test, y_test, pred_s)

    # 4) q+s linear regression
    lr_qs = LinearRegression()
    lr_qs.fit(train_df[["q_ai_aggregate", "s_sim"]].to_numpy(), y_train)
    pred_qs = lr_qs.predict(test_df[["q_ai_aggregate", "s_sim"]].to_numpy())
    write_pred(out_root / "q_s_lr", ad_test, y_test, pred_qs)

    print("Wrote baselines:")
    for name in ["mean_pred", "q_only_lr", "s_only_lr", "q_s_lr"]:
        print(f"  {out_root / name / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
