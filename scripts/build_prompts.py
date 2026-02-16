#!/usr/bin/env python3
"""
Build text prompts for AI video generation from Hussain annotations.
ROBUST VERSION: Recursively searches for annotation files.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_mapping_file(file_path):
    mapping = {}
    if not file_path.exists(): return mapping
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].strip('.').isdigit():
                    mapping[int(parts[0].strip('.'))] = {'description': parts[1].strip()}
    except: pass
    return mapping

def find_file_recursive(base_path, pattern):
    """Finds a file matching pattern anywhere in subdirectories."""
    # 1. Exact match in base
    if (base_path / pattern).exists():
        return base_path / pattern
    # 2. Recursive search
    for f in base_path.rglob(pattern):
        return f
    # 3. Fuzzy search
    keyword = pattern.split('_')[1] if '_' in pattern else pattern.replace('.json','')
    for f in base_path.rglob(f"*{keyword}*.json"):
        return f
    return None

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--climax_dir", default="data/raw/hussain_climax", help="Directory containing extracted *_real_6s.mp4 files.")
    ap.add_argument("--out_csv", default="data/metadata/ad_prompts.csv", help="Where to write prompts CSV.")
    ap.add_argument("--max_videos", type=int, default=0, help="Optional cap on number of prompts written (0 = no cap).")
    ap.add_argument("--min_p_real", type=float, default=0.5, help="Filter: keep only ads with P_real >= this threshold.")
    ap.add_argument("--annotations_root", default="data/raw/hussain", help="Root directory for Hussain annotation JSONs.")
    args = ap.parse_args()

    logger.info("Building prompts...")
    
    # 1. Setup Paths
    raw_dir = Path(args.annotations_root)
    climax_dir = Path(args.climax_dir)
    # First-time runs may not have extracted clips yet; don't fail.
    # If the directory does not exist, create it and fall back to annotation IDs.
    if not climax_dir.exists():
        climax_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Climax directory did not exist; created: {climax_dir}. Will fall back to annotation IDs.")
    
    # 2. Find Files (Robust Strategy)
    eff_path = find_file_recursive(raw_dir, "video_Effective_clean.json") or find_file_recursive(raw_dir, "video_Effective.json")
    topic_path = find_file_recursive(raw_dir, "video_Topics.json")
    sent_path = find_file_recursive(raw_dir, "video_Sentiments.json")
    
    if not eff_path:
        logger.error(f"Could not find Effective scores JSON under: {raw_dir}")
        sys.exit(1)
        
    logger.info(f"Found effective-score annotations: {eff_path}")

    # 3. Load Data
    with open(eff_path, 'r') as f: eff_data = json.load(f)
    
    # Optional mappings
    topic_data = {}
    if topic_path: 
        with open(topic_path, 'r') as f: topic_data = json.load(f)
        
    sent_data = {}
    if sent_path:
        with open(sent_path, 'r') as f: sent_data = json.load(f)

    # 4. Build Prompts
    # Prefer IDs that actually exist as extracted 6s clips; otherwise fall back to annotation IDs.
    target_ids = []
    for p in sorted(climax_dir.glob("*_real_6s.mp4")):
        name = p.name
        if name.endswith("_real_6s.mp4"):
            target_ids.append(name.replace("_real_6s.mp4", ""))

    if not target_ids:
        logger.warning("No *_real_6s.mp4 files found. Falling back to IDs from effective-score annotations JSON.")
        target_ids = list(map(str, eff_data.keys()))

    if args.max_videos and args.max_videos > 0:
        target_ids = target_ids[: int(args.max_videos)]

    logger.info(f"Generating prompts for {len(target_ids)} videos...")

    prompts = []
    for vid in target_ids:
        # Defaults
        topic = "commercial product"
        sentiment = "engaging"
        
        # Look up specifics
        if vid in topic_data: topic = f"Topic {topic_data[vid]}" # Simplified lookup
        if vid in sent_data: sentiment = f"Sentiment {sent_data[vid]}"
        
        # Get Score (Ground Truth)
        score = eff_data.get(vid, 3.0) # Default to mid-range

        # Apply P_real filtering (main benchmark uses P_real >= 0.5)
        try:
            p_real = (float(score) - 1.0) / 4.0
        except Exception:
            p_real = 0.0
        if args.min_p_real is not None and p_real < float(args.min_p_real):
            continue
        
        prompts.append({
            'ad_id': vid,
            'prompt': f"High quality {sentiment} advertisement for {topic}, 4k, cinematic",
            'effective_score': score,
            'p_real': p_real,
            'topic': topic,
            'sentiment': sentiment
        })

    # 5. Save
    df = pd.DataFrame(prompts)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} prompts to {out_path}")

if __name__ == "__main__":
    main()