#!/usr/bin/env python3
"""
Generates Table III: Comparative Analysis of Fusion Mechanisms.
Reads test_predictions.csv from 3 experiment folders.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

EXPERIMENTS = {
    "Baseline (Concat)": "experiments/baseline_concat",
    "Standard Attention": "experiments/baseline_attn",
    "Ours (Quality Gated)": "experiments/proposed_gated"
}

def compute_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    y_true = df['p_retained']
    y_pred = df['p_pred']
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r, _ = pearsonr(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'Pearson r': r}

def main():
    results = []
    print(f"{'Method':<25} | {'MAE':<8} | {'RMSE':<8} | {'Pearson r':<10}")
    print("-" * 60)
    
    for name, path in EXPERIMENTS.items():
        csv_file = os.path.join(path, "test_predictions.csv")
        metrics = compute_metrics(csv_file)
        
        if metrics:
            print(f"{name:<25} | {metrics['MAE']:.4f}   | {metrics['RMSE']:.4f}   | {metrics['Pearson r']:.4f}")
            results.append({
                "Method": name,
                "MAE": metrics['MAE'],
                "RMSE": metrics['RMSE'],
                "Pearson r": metrics['Pearson r']
            })
        else:
            print(f"{name:<25} | Not Found (Run training first)")

    # Generate LaTeX table
    print("\n[LaTeX Table III Body]")
    print(r"% Method & MAE & RMSE & Pearson r \\")
    best_mae = min(r['MAE'] for r in results) if results else 0
    
    for res in results:
        # Bold the best result
        mae_str = f"\\textbf{{{res['MAE']:.4f}}}" if res['MAE'] == best_mae else f"{res['MAE']:.4f}"
        print(f"{res['Method']} & {mae_str} & {res['RMSE']:.4f} & {res['Pearson r']:.4f} \\\\")
        print(r"\hline")

if __name__ == "__main__":
    main()