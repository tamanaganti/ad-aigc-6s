# scripts/generate_tables_and_figures.py
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_METHOD_ALIASES = {
    "concat": "Concatenation + MLP",
    "standard_attn": "Standard Attention",
    "quality_gated": "Quality-Gated Attention",
    "mean_pred": "Mean Prediction",
    "q_only_lr": "Linear [$\\bar{Q}_{ai}$]",
    "s_only_lr": "Linear [$S_{sim}$]",
    "q_s_lr": "Linear [$\\bar{Q}_{ai}, S_{sim}$]",
}

DEFAULT_QUALITY_COLUMNS = [
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_temporal_flicker",
    "vbench_sharpness_consistency",
]

SEED_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<seed>\d+)$")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(df: pd.DataFrame, abs_tol: float = 0.1) -> dict:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    within_abs = float(np.mean(np.abs(y_true - y_pred) <= abs_tol))

    return {
        "MAE": mae,
        "RMSE": rmse,
        f"Within_abs_{abs_tol:g}": within_abs,
    }


def load_predictions_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Map to y_true/y_pred/ad_id
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["ad_id", "id", "video_id", "sample_id"]:
            col_map[c] = "ad_id"
        if lc in ["y_true", "true", "target", "gt", "label", "p_retained"]:
            col_map[c] = "y_true"
        if lc in ["y_pred", "pred", "prediction", "output", "p_pred"]:
            col_map[c] = "y_pred"

    df = df.rename(columns=col_map)
    missing = {"y_true", "y_pred"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing {sorted(missing)}. Found: {list(df.columns)}. "
            "Expected either (y_true,y_pred) or (p_retained,p_pred)."
        )

    if "ad_id" not in df.columns:
        df["ad_id"] = np.arange(len(df), dtype=int)

    out = df[["ad_id", "y_true", "y_pred"]].copy()
    out["ad_id"] = out["ad_id"].astype(str)
    out["y_true"] = out["y_true"].astype(float)
    out["y_pred"] = out["y_pred"].astype(float)
    return out


def infer_method_key(path: Path) -> str:
    # Prefer folder name, else filename
    parent = path.parent.name.lower()
    stem = path.stem.lower()

    # common: concat_42, quality_gated_999, mean_pred
    for candidate in [parent, stem]:
        # normalize known patterns
        if candidate.startswith("baseline_"):
            candidate = candidate.replace("baseline_", "")
        if candidate.startswith("proposed_"):
            candidate = candidate.replace("proposed_", "")
        return candidate
    return parent


def method_base_and_seed(method_key: str):
    m = SEED_SUFFIX_RE.match(method_key)
    if m:
        return m.group("base"), int(m.group("seed"))
    return method_key, None


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def fit_oracle(dataset_df: pd.DataFrame, feature_mode: str, test_ids: set) -> pd.DataFrame:
    if feature_mode not in ["q_s", "p_q_s", "pq_ps"]:
        raise ValueError("feature_mode must be one of: q_s, p_q_s, pq_ps")

    required = {"ad_id", "p_retained", "q_ai_aggregate", "s_sim", "p_real"}
    missing = required - set(dataset_df.columns)
    if missing:
        raise ValueError(f"Dataset missing for oracle: {sorted(missing)}")

    df = dataset_df.copy()
    df["ad_id"] = df["ad_id"].astype(str)
    df = df[df["ad_id"].isin(test_ids)].copy()

    y_true = df["p_retained"].to_numpy(dtype=float)

    if feature_mode == "q_s":
        X = df[["q_ai_aggregate", "s_sim"]].to_numpy(dtype=float)
        label = "Linear Oracle [$\\bar{Q}_{ai}, S_{sim}$]"
    elif feature_mode == "p_q_s":
        X = df[["p_real", "q_ai_aggregate", "s_sim"]].to_numpy(dtype=float)
        label = "Linear Oracle [$P_{real}, \\bar{Q}_{ai}, S_{sim}$]"
    else:
        X = np.stack(
            [
                (df["p_real"] * df["q_ai_aggregate"]).to_numpy(dtype=float),
                (df["p_real"] * df["s_sim"]).to_numpy(dtype=float),
            ],
            axis=1,
        )
        label = "Linear Oracle [$P_{real}\\bar{Q}_{ai}, P_{real}S_{sim}$]"

    model = LinearRegression()
    model.fit(X, y_true)
    y_pred = model.predict(X)

    out = pd.DataFrame({"ad_id": df["ad_id"].to_numpy(), "y_true": y_true, "y_pred": y_pred})
    return out, label


def compute_correlations(dataset_df: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    df = dataset_df.copy()

    if target == "retention_factor":
        if "retention_factor" in df.columns:
            y = df["retention_factor"].astype(float)
        else:
            lam = float(df["lambda"].iloc[0]) if "lambda" in df.columns else 0.5
            y = lam * df["q_ai_aggregate"].astype(float) + (1.0 - lam) * df["s_sim"].astype(float)
    elif target == "p_retained":
        y = df["p_retained"].astype(float)
    elif target == "q_ai":
        y = df["q_ai_aggregate"].astype(float)
    else:
        raise ValueError("target must be one of: p_retained, retention_factor, q_ai")

    rows = []
    for c in columns:
        if c not in df.columns:
            rows.append({"metric": c, "pearson_r": np.nan, "spearman_rho": np.nan})
            continue
        x = df[c].astype(float)
        rows.append({
            "metric": c,
            "pearson_r": float(x.corr(y, method="pearson")),
            "spearman_rho": float(x.corr(y, method="spearman")),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csvs", nargs="+", required=True)
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--abs_tol", type=float, default=0.1)

    ap.add_argument("--include_oracle", action="store_true")
    ap.add_argument("--oracle_features", default="p_q_s", choices=["q_s", "p_q_s", "pq_ps"])

    ap.add_argument("--correlation_target", default="retention_factor", choices=["p_retained", "retention_factor", "q_ai"])
    ap.add_argument("--correlation_metrics", nargs="*", default=DEFAULT_QUALITY_COLUMNS)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = pd.read_csv(args.dataset_csv)
    dataset_df["ad_id"] = dataset_df["ad_id"].astype(str)

    # Load predictions and compute per-file metrics
    per_file_rows = []
    method_to_test_ids = None

    for p in args.pred_csvs:
        path = Path(p)
        pred_df = load_predictions_csv(path)

        method_key = infer_method_key(path)
        base_key, seed = method_base_and_seed(method_key)

        pretty = DEFAULT_METHOD_ALIASES.get(base_key, base_key)
        metrics = compute_metrics(pred_df, abs_tol=args.abs_tol)

        per_file_rows.append({
            "MethodKey": base_key,
            "Method": pretty,
            "Seed": seed if seed is not None else -1,
            **metrics
        })

        # Use first loaded file to define test IDs for oracle alignment
        if method_to_test_ids is None:
            method_to_test_ids = set(pred_df["ad_id"].astype(str).tolist())

    per_file_df = pd.DataFrame(per_file_rows)

    # Oracle row (deterministic, std=0)
    oracle_label = None
    if args.include_oracle:
        oracle_pred, oracle_label = fit_oracle(dataset_df, feature_mode=args.oracle_features, test_ids=method_to_test_ids)
        oracle_metrics = compute_metrics(oracle_pred, abs_tol=args.abs_tol)
        oracle_row = {
            "MethodKey": "oracle",
            "Method": oracle_label,
            "Seed": -1,
            **oracle_metrics
        }
        per_file_df = pd.concat([pd.DataFrame([oracle_row]), per_file_df], ignore_index=True)

    # Aggregate mean/std across seeds (Seed==-1 is deterministic)
    metric_cols = ["MAE", "RMSE", f"Within_abs_{args.abs_tol:g}"]

    agg_rows = []
    for (mkey, mname), g in per_file_df.groupby(["MethodKey", "Method"], sort=False):
        # deterministic: only one row or Seed == -1
        if (g["Seed"] == -1).all() or len(g) == 1:
            row = {"Method": mname}
            for c in metric_cols:
                row[f"{c}_mean"] = float(g[c].iloc[0])
                row[f"{c}_std"] = 0.0
            agg_rows.append(row)
        else:
            row = {"Method": mname}
            for c in metric_cols:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=0))
            agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)

    # Write CSVs
    per_file_df.to_csv(out_dir / "table_main_results_per_seed.csv", index=False)
    agg_df.to_csv(out_dir / "table_main_results_mean_std.csv", index=False)

    # Write LaTeX mean±std table
    abs_col = f"Within_abs_{args.abs_tol:g}"
    lines = []
    lines.append("\\begin{tabular}{|l|c|c|c|}")
    lines.append("\\hline")
    lines.append(f"Method & MAE $\\downarrow$ & RMSE $\\downarrow$ & Within $\\pm {args.abs_tol:.2f}$ $\\uparrow$ \\\\")
    lines.append("\\hline")
    for _, r in agg_df.iterrows():
        method = latex_escape(str(r["Method"]))
        mae = float(r["MAE_mean"]); mae_s = float(r["MAE_std"])
        rmse = float(r["RMSE_mean"]); rmse_s = float(r["RMSE_std"])
        within = float(r[f"{abs_col}_mean"]); within_s = float(r[f"{abs_col}_std"])
        lines.append(f"{method} & {mae:.4f} $\\pm$ {mae_s:.4f} & {rmse:.4f} $\\pm$ {rmse_s:.4f} & {within:.4f} $\\pm$ {within_s:.4f} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    (out_dir / "table_main_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # Correlations table
    corr_df = compute_correlations(dataset_df, target=args.correlation_target, columns=args.correlation_metrics)
    corr_df.to_csv(out_dir / "table_quality_correlations.csv", index=False)

    print(f"Wrote: {out_dir / 'table_main_results_per_seed.csv'}")
    print(f"Wrote: {out_dir / 'table_main_results_mean_std.csv'}")
    print(f"Wrote: {out_dir / 'table_main_results.tex'}")
    print(f"Wrote: {out_dir / 'table_quality_correlations.csv'}")


if __name__ == "__main__":
    main()
