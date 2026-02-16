#!/usr/bin/env python3
"""
Generate AI video variants (Multi-GPU Sharding Mode).
Target: 5 Variants per Ad x 3400 Ads = ~17,000 Pairs.
Supports splitting workload across multiple GPUs.
"""

from __future__ import annotations

import torch
import os
import pandas as pd
import cv2
import numpy as np
import gc
import argparse
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import logging
from pathlib import Path
from typing import List

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# At 7 fps: 6 seconds = 42 frames.
# We generate two 3-second chunks of 21 frames each and concatenate them.
# Important: If we drop an overlapping seam frame (frames2[1:]) we get 41 frames (5.86s).
# For an exact 6.0s output at 7 fps, we keep all 42 frames (frames1 + frames2).
FPS = 7
FRAMES_PER_CHUNK = 21

DEFAULT_PILOT_STEPS = [10, 20, 30, 40, 50]
QUALITY_NAMES = {10: "LQ", 20: "MQ", 30: "MQ+", 40: "HQ", 50: "HQ+"}

def generate_ads_svd(input_csv: str, device_id: int, *, mode: str, steps_list: List[int], output_dir: str, log_csv: str):
    logger.info(f"GPU {device_id} STARTING | mode={mode} | input={input_csv}")

    if not os.path.exists(input_csv):
        logger.error(f"Input file {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"GPU {device_id}: Processing {len(df)} ads")

    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.to(f"cuda:{device_id}")
        logger.info(f"Model loaded on GPU {device_id}")
    except Exception as e:
        logger.error(f"Model load failed on GPU {device_id}: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Store results for this shard
    results = []
    log_path = log_csv

    for idx, row in df.iterrows():
        ad_id = str(row['ad_id'])
        real_path = f"data/raw/hussain_climax/{ad_id}_real_6s.mp4"

        if not os.path.exists(real_path):
            continue

        try:
            cap = cv2.VideoCapture(real_path)
            ret, frame = cap.read()
            cap.release()
            if not ret: continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((1024, 576))
            # Save reference frame once
            ref_path = os.path.join(output_dir, f"{ad_id}_real_frame.png")
            if not os.path.exists(ref_path):
                 img.save(ref_path)
        except: continue

        for steps in steps_list:
            if mode == "main":
                # Main benchmark uses the 10-step setting selected by pilot analysis.
                variant_id = f"{ad_id}_optimal" if int(steps) == 10 else f"{ad_id}_q{int(steps)}"
            else:
                variant_id = f"{ad_id}_q{int(steps)}"

            video_path = os.path.join(output_dir, f"{variant_id}_6s.mp4")

            # Skip if exists
            if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                continue

            try:
                # Cleanup
                torch.cuda.empty_cache()
                gc.collect()

                base_seed = 42 + steps * 10
                logger.info(f"[GPU {device_id}] Generating {variant_id} ({steps} steps)...")

                gen1 = torch.Generator(device=f"cuda:{device_id}").manual_seed(int(base_seed))
                gen2 = torch.Generator(device=f"cuda:{device_id}").manual_seed(int(base_seed) + 1)

                frames1 = pipe(
                    img,
                    decode_chunk_size=8,
                    num_frames=FRAMES_PER_CHUNK,
                    generator=gen1,
                    num_inference_steps=int(steps),
                ).frames[0]
                frames2 = pipe(
                    frames1[-1],
                    decode_chunk_size=8,
                    num_frames=FRAMES_PER_CHUNK,
                    generator=gen2,
                    num_inference_steps=int(steps),
                ).frames[0]

                # Keep all frames to produce an exact 6.0s clip at 7 fps (42 frames)
                export_to_video(frames1 + frames2, video_path, fps=FPS)
                
                # Log Success
                results.append({
                    'ad_id': ad_id,
                    'variant_id': variant_id,
                    'inference_steps': int(steps),
                    'quality_level': int(steps),
                    'quality_name': QUALITY_NAMES.get(int(steps), f"Q{int(steps)}"),
                    'status': 'success',
                    'real_path': real_path,
                    'ai_path': video_path
                })

            except Exception as e:
                logger.error(f"[GPU {device_id}] Failed {variant_id}: {e}")
                results.append({
                    'ad_id': ad_id,
                    'variant_id': variant_id,
                    'inference_steps': int(steps),
                    'status': 'failed',
                    'error': str(e)
                })

        # Save logs progressively (every 10 ads) or at end
        if len(results) >= 50 or idx == len(df) - 1:
            df_log = pd.DataFrame(results)
            # Append mode if file exists, else write
            mode = 'a' if os.path.exists(log_path) else 'w'
            header = not os.path.exists(log_path)
            df_log.to_csv(log_path, mode=mode, header=header, index=False)
            results = [] # Clear buffer

    logger.info(f"GPU {device_id} Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV with at least an ad_id column (e.g., data/metadata/ad_prompts.csv).")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot", help="pilot=multi-step ablation, main=10-step benchmark")
    parser.add_argument("--steps", type=str, default="", help="Comma-separated inference steps override (e.g., '10,20,30'). If empty, defaults are used.")
    parser.add_argument("--output_dir", type=str, default="", help="Where to write generated AI videos.")
    parser.add_argument("--log_csv", type=str, default="", help="Where to write generation log CSV (append mode).")
    args = parser.parse_args()

    if args.steps.strip():
        steps_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    else:
        steps_list = [10] if args.mode == "main" else list(DEFAULT_PILOT_STEPS)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "data/videos/ai_ads_main_experiment" if args.mode == "main" else "data/videos/pilot_study_ablation"

    if args.log_csv:
        log_csv = args.log_csv
    else:
        log_csv = "data/metadata/main_experiment_generation_log.csv" if args.mode == "main" else f"data/metadata/svd_log_gpu{args.device}.csv"

    generate_ads_svd(
        args.input,
        args.device,
        mode=args.mode,
        steps_list=steps_list,
        output_dir=output_dir,
        log_csv=log_csv,
    )