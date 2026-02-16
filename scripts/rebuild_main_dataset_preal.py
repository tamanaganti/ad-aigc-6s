#!/usr/bin/env python3
"""
Rebuild main dataset from Hussain effective_score: p_real = (effective_score - 1) / 4.
Applies selection filter (default [0.75, 0.85]) and generates distribution report.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

HUSSAIN_JSON = "data/raw/hussain/video/cleaned_result/video_Effective_clean.json"
DEFAULT_INPUT_CSV = "data/metadata/dataset_main_shard_0.csv"
DEFAULT_OUTPUT_CSV = "data/metadata/ad_aigc_6s_main_experiment.csv"

def load_p_real_map():
    """Load map {ad_id: p_real} from Hussain JSON."""
    if not os.path.exists(HUSSAIN_JSON):
        print(f"Hussain annotations not found: {HUSSAIN_JSON}")
        return {}
    
    with open(HUSSAIN_JSON, "r") as f:
        raw = json.load(f)
        
    out = {}
    for k, v in raw.items():
        try:
            score = int(v)
            if 1 <= score <= 5:
                # Map 1..5 -> 0.0, 0.25, 0.5, 0.75, 1.0
                out[str(k).strip()] = (score - 1) / 4.0
        except (TypeError, ValueError):
            pass
    print(f"Loaded {len(out)} scores from JSON.")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--low", type=float, default=0.50, help="Lower bound for p_real (inclusive)")
    parser.add_argument("--high", type=float, default=1.00, help="Upper bound for p_real (inclusive)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}")
        return

    # 1. Load Data & Scores
    df = pd.read_csv(args.input)
    p_real_map = load_p_real_map()
    
    if not p_real_map:
        print("No JSON map loaded. Relying on existing 'p_real' column if present.")
    
    # 2. Update p_real column (Single Source of Truth)
    def get_score(row):
        # Try direct string match
        ad_id = str(row['ad_id']).strip()
        if ad_id in p_real_map:
            return p_real_map[ad_id]
        
        # Try float->int->string match (e.g. 1001.0 -> "1001")
        try:
            int_id = str(int(float(row['ad_id'])))
            if int_id in p_real_map:
                return p_real_map[int_id]
        except:
            pass
            
        # Fallback to existing column
        if 'p_real' in row:
            return row['p_real']
        return np.nan

    df['p_real'] = df.apply(get_score, axis=1)
    
    # 3. Print Distribution BEFORE Filter
    print("\n" + "="*60)
    print("DISTRIBUTION BEFORE FILTERING")
    print("="*60)
    print(df['p_real'].value_counts(dropna=False).sort_index())
    
    mean_before = df['p_real'].mean()
    print(f"\nOverall Mean: {mean_before:.4f}")
    
    # 4. Apply Filter
    print("\n" + "="*60)
    print(f"APPLYING FILTER: [{args.low}, {args.high}]")
    print("="*60)
    
    df_filtered = df[
        (df['p_real'] >= args.low) & 
        (df['p_real'] <= args.high)
    ].copy()
    
    if len(df_filtered) == 0:
        print("Filter resulted in 0 rows. Check your bounds.")
        return

    # 5. Print Distribution AFTER Filter
    print(df_filtered['p_real'].value_counts().sort_index())
    
    mean_after = df_filtered['p_real'].mean()
    std_after = df_filtered['p_real'].std()
    
    print(f"\nFinal Count: {len(df_filtered)} rows (dropped {len(df) - len(df_filtered)})")
    print(f"Final Mean:  {mean_after:.4f}")
    print(f"Final Std:   {std_after:.4f}")
    
    # 6. Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()