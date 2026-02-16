#!/usr/bin/env python3
"""
Ablation Study Analysis (Table II & Figure 3).
Consumes 'quality_metrics_pilot.csv'.
Robustly extracts 'inference_steps' from column OR 'variant_id' string.
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

# --- Constants ---
PILOT_METRICS_PATH = "data/metadata/quality_metrics_pilot.csv"
OUTPUT_DIR = "outputs/paper_artifacts"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

QUALITY_NAMES = {10: "LQ", 20: "MQ", 30: "MQ+", 40: "HQ", 50: "HQ+"}
COMPUTE_COSTS = {10: 1.0, 20: 2.0, 30: 3.0, 40: 4.0, 50: 5.0}

def extract_steps_from_variant(variant_id):
    """Extracts step count from variant_id like 'ad123_q20' -> 20."""
    try:
        # Look for _q10, _q20, etc.
        match = re.search(r'_q(\d+)', str(variant_id))
        if match:
            return int(match.group(1))
        # Fallback: sometimes logged as 'steps_10'
        match = re.search(r'steps_(\d+)', str(variant_id))
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def calculate_q_ai(df):
    """Calculate the official 33-dimensional Q_AI score."""
    vbench_cols = [c for c in df.columns if c.startswith('vbench_')]
    eval_cols = [c for c in df.columns if c.startswith('evalcrafter_')]
    
    # 60% Visual Quality (VBench), 40% Temporal (EvalCrafter)
    v_mean = df[vbench_cols].mean(axis=1) if vbench_cols else 0.5
    e_mean = df[eval_cols].mean(axis=1) if eval_cols else 0.5
    
    df['q_ai'] = 0.6 * v_mean + 0.4 * e_mean
    return df

def main():
    print("--- ABLATION STUDY ANALYSIS (TABLE II & FIGURE 3) ---")
    
    # 1. Load Data
    # Support loading the uploaded file name if strictly needed, but better to rename it 
    # to the standard path: data/metadata/quality_metrics_pilot.csv
    if not os.path.exists(PILOT_METRICS_PATH):
        print(f"Pilot metrics not found: {PILOT_METRICS_PATH}")
        return

    df = pd.read_csv(PILOT_METRICS_PATH)
    print(f"Loaded {len(df)} pilot samples.")

    # 2. Robust Step Extraction
    if 'inference_steps' not in df.columns or df['inference_steps'].isna().all():
        print("'inference_steps' missing or empty. Extracting from 'variant_id'...")
        df['inference_steps'] = df['variant_id'].apply(extract_steps_from_variant)
    
    # Drop rows where steps couldn't be determined
    df = df.dropna(subset=['inference_steps'])
    df['inference_steps'] = df['inference_steps'].astype(int)
    print(f"Valid samples with steps: {len(df)}")

    # 3. Compute Q_ai
    df = calculate_q_ai(df)

    # 4. Group by Steps
    stats = []
    levels = sorted([s for s in df['inference_steps'].unique() if s in COMPUTE_COSTS])
    
    for level in levels:
        subset = df[df['inference_steps'] == level]
        if len(subset) == 0: continue
        
        cost = COMPUTE_COSTS.get(level, level/10.0)
        q_score = subset['q_ai'].mean()
        
        stats.append({
            'Quality Level': QUALITY_NAMES.get(level, f"Steps-{level}"),
            'Inference Steps': int(level),
            'Compute Cost': cost,
            'Samples': len(subset),
            'Q_AI': q_score,
            'Efficiency': q_score / cost 
        })

    res_df = pd.DataFrame(stats)

    # 5. Print Table II
    print("\n[LaTeX Table II Body]")
    print(r"% Level & Steps & Cost & Samples & Ratio \\")
    for _, row in res_df.iterrows():
        print(f"{row['Quality Level']} & {row['Inference Steps']} & {row['Compute Cost']:.1f}x & {row['Samples']} & {row['Efficiency']:.3f} \\\\")
        print(r"\hline")
        
    # 6. Plot Figure 3
    fig, ax1 = plt.subplots(figsize=(8, 5))
    steps = res_df['Inference Steps']
    quality = res_df['Q_AI']
    cost = res_df['Compute Cost']
    
    color = 'tab:blue'
    ax1.set_xlabel('Inference Steps')
    ax1.set_ylabel('Aggregate Quality ($Q_{ai}$)', color=color)
    ax1.plot(steps, quality, marker='o', color=color, linewidth=2, label='Quality')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Auto-scale Y to look "zoomed in" but safe
    y_min, y_max = quality.min(), quality.max()
    margin = (y_max - y_min) * 0.5
    ax1.set_ylim(max(0, y_min - margin), min(1, y_max + margin))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative Compute Cost', color=color)
    ax2.bar(steps, cost, color=color, alpha=0.3, width=4, label='Cost')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Quality vs. Efficiency Trade-off (Ablation)")
    plt.tight_layout()
    
    out_path = f"{OUTPUT_DIR}/figure_3_ablation_elbow.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved figure to {out_path}")

if __name__ == "__main__":
    main()