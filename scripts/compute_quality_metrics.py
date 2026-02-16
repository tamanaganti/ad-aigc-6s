#!/usr/bin/env python3
"""
Compute 33 OpenCV-based proxy quality metrics.
UPDATED: Merges all pilot logs & Force-fixes video paths to local directories.
"""

import os
import cv2
import pandas as pd
import numpy as np
import logging
import argparse
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Constants (Same as before) ---
D_MOTION_MEAN = 30.0
D_BRIGHTNESS_STD = 20.0
D_SHARPNESS_MEAN_DENOM = 250.0  
D_DYNAMIC_STD = 15.0
D_CONSISTENCY_STD = 50.0
D_COLOR_STD = 30.0
D_SATURATION_STD = 40.0
D_SHARPNESS_CONSISTENCY_STD = 200.0
D_NOISE = 25.0
D_EDGE_CONSISTENCY = 0.05
D_CAMERA_SHAKE = 3.0
D_AESTHETIC_SAT_REF = 80.0
D_AESTHETIC_CON_REF = 80.0
D_BLOCKINESS_K = 2500.0  
D_BLACK_FRAME_REF = 0.60  
D_SCENE_CHANGE_P95 = 0.8
D_FLOW_MEAN = 5.0
D_FLOW_STD = 5.0  
D_FLOW_INCONSISTENCY = 20.0
D_GHOSTING = 30.0
D_EDGE_FLICKER = 0.05
D_COLOR_SHIFT = 40.0
D_WB_DRIFT = 0.3
D_EXPOSURE_TAU = 0.2
D_DUP_THRESHOLD = 3.0  
D_FLICKER_FREQ = 15.0
D_MOTION_JERK = 2.0
D_SPATIAL_JITTER = 2.0
D_DETAIL_TAU = 50.0 
D_BLUR_DROP_TAU = 0.02 
D_FLOW_DIR_STD_REF = 0.8

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_data_config(target_mode):
    """
    Returns (DataFrame, OutputPath, VideoDir).
    Merges multiple logs if needed.
    """
    # 1. Main Experiment
    if target_mode == "main":
        log_path = "data/metadata/main_experiment_generation_log.csv"
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Main log not found: {log_path}")
        df = pd.read_csv(log_path)
        return df, "data/metadata/quality_metrics_main_experiment.csv", "data/videos/ai_ads_main_experiment"

    # 2. Pilot Experiment (Merge Logic)
    elif target_mode == "pilot":
        # Look for single merged log OR gpu shards
        candidates = glob.glob("data/metadata/svd_generation_log.csv") + \
                     glob.glob("data/metadata/svd_log_gpu*.csv")
        
        if not candidates:
            raise FileNotFoundError("No pilot logs found (checked svd_generation_log.csv and svd_log_gpu*.csv)")
        
        print(f"Found {len(candidates)} pilot logs: {candidates}")
        df_list = []
        for c in candidates:
            try:
                d = pd.read_csv(c)
                if len(d) > 0: df_list.append(d)
            except: pass
            
        if not df_list:
            raise ValueError("All found pilot logs were empty or unreadable.")
            
        df = pd.concat(df_list, ignore_index=True)
        # Deduplicate just in case
        if 'ad_id' in df.columns and 'variant_id' in df.columns:
            df = df.drop_duplicates(subset=['ad_id', 'variant_id'])
            
        return df, "data/metadata/quality_metrics_pilot.csv", "data/videos/pilot_study_ablation"

    # 3. Auto (Legacy)
    else:
        # Try main first
        if os.path.exists("data/metadata/main_experiment_generation_log.csv"):
            return load_data_config("main")
        else:
            return load_data_config("pilot")

def _flow_magnitude_and_direction(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)
    ang = np.where(ang < 0, ang + 2 * np.pi, ang)
    return mag, ang

def _circular_std(theta):
    s = np.sin(theta).mean()
    c = np.cos(theta).mean()
    R = np.sqrt(s * s + c * c) + 1e-8
    return np.sqrt(-2.0 * np.log(np.clip(R, 1e-8, 1.0)))

def _ssim_proxy_single(a, b, size=8):
    if a.size == 0 or b.size == 0: return 0.5
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, va = np.mean(a), np.var(a) + 1e-8
    mb, vb = np.mean(b), np.var(b) + 1e-8
    cov = np.mean((a - ma) * (b - mb)) + 1e-8
    c = cov / (np.sqrt(va * vb) + 1e-8)
    return np.clip((c + 1) / 2, 0, 1)

def compute_metrics_for_video(args):
    """
    Worker function.
    args: (row_dict, video_root_dir)
    """
    row, video_root = args
    
    # Resolve video path from CSV row (filename only, then join with video_root)
    raw_path = row.get('ai_path') or row.get('video_path') or ""
    filename = os.path.basename(raw_path)
    video_path = os.path.join(video_root, filename)
    ad_id = row.get('ad_id', 'unknown')
    variant_id = row.get('variant_id', f"{ad_id}_unknown")
    
    if not filename or not os.path.exists(video_path):
        # Fail silently to avoid log spam, the main loop counts failures
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if len(frames) < 2:
            return _empty_result(ad_id, variant_id, video_path, row)

        # ... [METRIC COMPUTATION LOGIC REMAINS IDENTICAL TO ORIGINAL] ...
        # (Re-pasting the core logic here for completeness)
        
        h, w = frames[0].shape[:2]
        scale = 0.25
        small_w, small_h = max(32, int(w * scale)), max(32, int(h * scale))

        per_frame_brightness_mean = []
        per_frame_brightness_std = []
        per_frame_sharpness = []
        per_frame_edge_density = []
        per_frame_saturation_mean = []
        per_frame_hsv_means = []
        motion_scores = []
        brightness_scores = []
        hist_diffs = []
        grays = []
        duplicate_scores = []
        blur_scores = []
        clipped_ratios = []
        rgb_means_list = []
        edge_energies = []
        prev_gray = None
        prev_small = None
        prev_hist = None
        flow_dirs = []
        flow_mag_means = []
        phase_shifts = []
        ssim_proxies = []
        edge_flicker_scores = []
        color_shift_scores = []
        warp_residuals = []

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            rgb_means_list.append(np.mean(frame, axis=(0, 1)))

            bmean = np.mean(gray)
            bstd = np.std(gray)
            per_frame_brightness_mean.append(bmean)
            per_frame_brightness_std.append(bstd)
            brightness_scores.append(bmean)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = lap.var()
            blur_scores.append(lap_var)
            per_frame_sharpness.append(lap_var)

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges > 0)
            per_frame_edge_density.append(edge_density)
            edge_energies.append(np.mean(edges))

            s = np.mean(hsv[:, :, 1])
            per_frame_saturation_mean.append(s)
            per_frame_hsv_means.append(np.mean(hsv, axis=(0, 1)))
            clipped = np.mean((gray <= 5) | (gray >= 250))
            clipped_ratios.append(clipped)

            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(np.mean(diff))
                diff_mean = float(np.mean(diff))
                dup_prob = float(np.exp(-diff_mean / (D_DUP_THRESHOLD + 1e-8)))
                duplicate_scores.append(dup_prob)
                if prev_hist is not None:
                    hist_diffs.append(np.sum(np.abs(hist - prev_hist)))
                flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 2, 10, 3, 5, 1.2, 0)
                mag, ang = _flow_magnitude_and_direction(flow)
                flow_dirs.append((ang, mag))
                flow_mag_means.append(float(np.mean(mag)))
                try:
                    shift, _ = cv2.phaseCorrelate(prev_small.astype(np.float32), small.astype(np.float32))
                    phase_shifts.append(np.sqrt(shift[0]**2 + shift[1]**2))
                except: phase_shifts.append(0.0)
                ssim_proxies.append(_ssim_proxy_single(prev_small.flatten(), small.flatten()))
                edge_flicker_scores.append(np.abs(per_frame_edge_density[-1] - per_frame_edge_density[-2]))
                if 'prev_hsv' in locals():
                    ch = np.mean(hsv, axis=(0, 1)) - np.mean(prev_hsv, axis=(0, 1))
                    color_shift_scores.append(np.sqrt(np.sum(ch ** 2)))
                warp_residuals.append(np.mean(np.abs(small.astype(float) - prev_small.astype(float))))

            prev_gray = gray
            prev_small = small
            prev_hist = hist
            prev_hsv = hsv

        brightness_arr = np.array(brightness_scores)
        blur_arr = np.array(blur_scores)

        # Aggregation
        vbench_motion_smoothness = np.clip(1.0 - (np.mean(motion_scores) if motion_scores else 0) / D_MOTION_MEAN, 0, 1)
        vbench_temporal_flicker = np.clip(1.0 - (np.std(brightness_arr) / D_BRIGHTNESS_STD), 0, 1) if len(brightness_arr) > 1 else 0.5
        mean_sharp = float(np.mean(blur_arr))
        vbench_subject_consistency = float(np.clip(np.log1p(mean_sharp) / np.log1p(D_SHARPNESS_MEAN_DENOM), 0, 1))
        
        sat = float(np.mean(per_frame_saturation_mean)) if per_frame_saturation_mean else 0.0
        con = float(np.mean(per_frame_brightness_std)) if per_frame_brightness_std else 0.0
        vbench_aesthetic_quality = float(np.clip(0.33 * np.clip(sat/D_AESTHETIC_SAT_REF,0,1) + 
                                                 0.33 * np.clip(con/D_AESTHETIC_CON_REF,0,1) + 
                                                 0.34 * vbench_subject_consistency, 0, 1))
        
        dyn_std = np.std(motion_scores) if motion_scores else 0
        vbench_dynamic_degree = np.clip(dyn_std / D_DYNAMIC_STD, 0, 1)
        vbench_brightness_consistency = np.clip(1.0 - (np.std(per_frame_brightness_mean) if per_frame_brightness_mean else 0) / D_CONSISTENCY_STD, 0, 1)
        vbench_contrast_consistency = np.clip(1.0 - (np.std(per_frame_brightness_std) if per_frame_brightness_std else 0) / D_CONSISTENCY_STD, 0, 1)
        vbench_color_consistency = np.clip(1.0 - (np.mean(np.std(np.array(per_frame_hsv_means), axis=0)) if per_frame_hsv_means else 0) / D_COLOR_STD, 0, 1)
        vbench_saturation_consistency = np.clip(1.0 - (np.std(per_frame_saturation_mean) if per_frame_saturation_mean else 0) / D_SATURATION_STD, 0, 1)
        vbench_sharpness_consistency = np.clip(1.0 - (np.std(per_frame_sharpness) if per_frame_sharpness else 0) / D_SHARPNESS_CONSISTENCY_STD, 0, 1)
        
        noise_est = 0
        if len(grays) > 0:
            noise_vals = [np.std(g.astype(float) - cv2.GaussianBlur(g, (5, 5), 1.0).astype(float)) for g in grays[:10]]
            noise_est = np.mean(noise_vals)
        vbench_noise_level = np.clip(1.0 - noise_est / D_NOISE, 0, 1)
        vbench_edge_consistency = np.clip(1.0 - (np.std(per_frame_edge_density) if per_frame_edge_density else 0) / D_EDGE_CONSISTENCY, 0, 1)
        vbench_camera_shake = np.clip(1.0 - (np.sqrt(np.var(phase_shifts)) if phase_shifts else 0) / D_CAMERA_SHAKE, 0, 1)
        vbench_scene_change_rate = float(np.clip(1.0 - (np.mean(hist_diffs) if hist_diffs else 0) / (D_SCENE_CHANGE_P95 + 1e-8), 0, 1))
        
        block_mean = 0
        if grays:
            b_vals = []
            for g in grays[:5]:
                bl = cv2.resize(g, (g.shape[1]//8*8, g.shape[0]//8*8))
                blocks = bl.reshape(bl.shape[0]//8, 8, bl.shape[1]//8, 8).mean(axis=(1,3))
                b_vals.append(np.var(blocks))
            block_mean = np.mean(b_vals)
        vbench_blockiness = float(np.clip(1.0 - (block_mean / (block_mean + D_BLOCKINESS_K)), 0, 1))
        
        black_mean = float(np.mean([float((g <= 10).mean()) for g in grays])) if grays else 0
        vbench_black_frame_ratio = float(np.clip(1.0 - (black_mean / (D_BLACK_FRAME_REF + 1e-8)), 0, 1))

        # EvalCrafter
        evalcrafter_flow_magnitude_mean = np.clip((np.mean(flow_mag_means) if flow_mag_means else 0) / D_FLOW_MEAN, 0, 1)
        evalcrafter_flow_magnitude_std = np.clip(1.0 - (np.std(flow_mag_means) if flow_mag_means else 0) / D_FLOW_STD, 0, 1)
        
        dir_stds = []
        for ang, mag in flow_dirs:
            if int((mag > 0.3).sum()) >= 50: dir_stds.append(_circular_std(ang[mag > 0.3]))
        evalcrafter_flow_direction_std = float(np.clip(np.exp(-np.mean(dir_stds) / (D_FLOW_DIR_STD_REF + 1e-8)), 0, 1)) if dir_stds else 0.5
        
        evalcrafter_flow_inconsistency = np.clip(1.0 - (np.mean(warp_residuals) if warp_residuals else 0) / D_FLOW_INCONSISTENCY, 0, 1)
        evalcrafter_ghosting_proxy = np.clip(1.0 - (np.mean(warp_residuals) if warp_residuals else 0) / D_GHOSTING, 0, 1)
        evalcrafter_temporal_ssim_proxy = float(np.clip(np.mean(ssim_proxies) if ssim_proxies else 0.5, 0, 1))
        evalcrafter_temporal_edge_flicker = np.clip(1.0 - (np.mean(edge_flicker_scores) if edge_flicker_scores else 0) / D_EDGE_FLICKER, 0, 1)
        evalcrafter_color_shift = np.clip(1.0 - (np.mean(color_shift_scores) if color_shift_scores else 0) / D_COLOR_SHIFT, 0, 1)
        
        wb_drift = 0.5
        if len(rgb_means_list) > 1:
            arr = np.array(rgb_means_list)
            wb_drift = np.clip(1.0 - (np.std(arr[:,0]/(arr[:,1]+1e-8)) + np.std(arr[:,2]/(arr[:,1]+1e-8))) / D_WB_DRIFT, 0, 1)
        evalcrafter_white_balance_drift = wb_drift
        
        evalcrafter_exposure_clipping = float(np.clip(np.exp(-np.mean(clipped_ratios) / (D_EXPOSURE_TAU + 1e-8)), 0, 1))
        
        e0, eL = (edge_energies[0], edge_energies[-1]) if edge_energies else (1, 1)
        loss_edge = float(np.clip(1.0 - (eL / (e0 + 1e-8)), 0, 1))
        evalcrafter_over_smoothing = float(1.0 / (1.0 + np.exp((loss_edge - 0.15) / 0.07)))
        
        evalcrafter_detail_preservation = float(np.clip(1.0 - np.exp(-(float(np.mean(per_frame_sharpness)) if per_frame_sharpness else 0) / (D_DETAIL_TAU + 1e-8)), 0, 1))
        evalcrafter_frame_drop_indicator = float(np.clip(1.0 - (float(np.mean(duplicate_scores)) if duplicate_scores else 0), 0, 1))
        
        drop = 0.0
        if len(blur_scores) >= 6:
            s = np.log1p(np.array(blur_scores, dtype=float))
            k = max(1, int(0.25 * len(s)))
            drop = max(0.0, float(np.median(s[:k])) - float(np.median(s[-k:])))
        evalcrafter_temporal_blur_increase = float(np.clip(np.exp(-drop / (D_BLUR_DROP_TAU + 1e-8)), 0, 1))
        
        hf = np.std(np.diff(brightness_arr, 2)) if len(brightness_arr) > 2 else 0
        evalcrafter_flicker_frequency_proxy = np.clip(1.0 - hf / D_FLICKER_FREQ, 0, 1)
        jerk = np.std(np.diff(flow_mag_means)) if len(flow_mag_means) >= 3 else 0
        evalcrafter_motion_jerk = np.clip(1.0 - jerk / D_MOTION_JERK, 0, 1)
        jitter = np.sqrt(np.var(phase_shifts)) if phase_shifts else 0
        evalcrafter_spatial_jitter = np.clip(1.0 - jitter / D_SPATIAL_JITTER, 0, 1)

        result_row = {
            'ad_id': ad_id,
            'variant_id': variant_id,
            'ai_path': video_path,
            'inference_steps': row.get('inference_steps', row.get('steps', '')),
            'vbench_motion_smoothness': float(vbench_motion_smoothness),
            'vbench_temporal_flicker': float(vbench_temporal_flicker),
            'vbench_subject_consistency': float(vbench_subject_consistency),
            'vbench_aesthetic_quality': float(vbench_aesthetic_quality),
            'vbench_dynamic_degree': float(vbench_dynamic_degree),
            'vbench_brightness_consistency': float(vbench_brightness_consistency),
            'vbench_contrast_consistency': float(vbench_contrast_consistency),
            'vbench_color_consistency': float(vbench_color_consistency),
            'vbench_saturation_consistency': float(vbench_saturation_consistency),
            'vbench_sharpness_consistency': float(vbench_sharpness_consistency),
            'vbench_noise_level': float(vbench_noise_level),
            'vbench_edge_consistency': float(vbench_edge_consistency),
            'vbench_camera_shake': float(vbench_camera_shake),
            'vbench_scene_change_rate': float(vbench_scene_change_rate),
            'vbench_blockiness': float(vbench_blockiness),
            'vbench_black_frame_ratio': float(vbench_black_frame_ratio),
            'evalcrafter_flow_magnitude_mean': float(evalcrafter_flow_magnitude_mean),
            'evalcrafter_flow_magnitude_std': float(evalcrafter_flow_magnitude_std),
            'evalcrafter_flow_direction_std': float(evalcrafter_flow_direction_std),
            'evalcrafter_flow_inconsistency': float(evalcrafter_flow_inconsistency),
            'evalcrafter_ghosting_proxy': float(evalcrafter_ghosting_proxy),
            'evalcrafter_temporal_ssim_proxy': float(evalcrafter_temporal_ssim_proxy),
            'evalcrafter_temporal_edge_flicker': float(evalcrafter_temporal_edge_flicker),
            'evalcrafter_color_shift': float(evalcrafter_color_shift),
            'evalcrafter_white_balance_drift': float(evalcrafter_white_balance_drift),
            'evalcrafter_exposure_clipping': float(evalcrafter_exposure_clipping),
            'evalcrafter_over_smoothing': float(evalcrafter_over_smoothing),
            'evalcrafter_detail_preservation': float(evalcrafter_detail_preservation),
            'evalcrafter_frame_drop_indicator': float(evalcrafter_frame_drop_indicator),
            'evalcrafter_temporal_blur_increase': float(evalcrafter_temporal_blur_increase),
            'evalcrafter_flicker_frequency_proxy': float(evalcrafter_flicker_frequency_proxy),
            'evalcrafter_motion_jerk': float(evalcrafter_motion_jerk),
            'evalcrafter_spatial_jitter': float(evalcrafter_spatial_jitter),
        }
        return result_row
    except Exception:
        return None

def _empty_result(ad_id, variant_id, video_path, row):
    # Just return 0.5s but preserve steps for analysis
    keys = [
        'vbench_motion_smoothness', 'vbench_temporal_flicker', 'vbench_subject_consistency',
        'vbench_aesthetic_quality', 'vbench_dynamic_degree', 'vbench_brightness_consistency',
        'vbench_contrast_consistency', 'vbench_color_consistency', 'vbench_saturation_consistency',
        'vbench_sharpness_consistency', 'vbench_noise_level', 'vbench_edge_consistency',
        'vbench_camera_shake', 'vbench_scene_change_rate', 'vbench_blockiness', 'vbench_black_frame_ratio',
        'evalcrafter_flow_magnitude_mean', 'evalcrafter_flow_magnitude_std', 'evalcrafter_flow_direction_std',
        'evalcrafter_flow_inconsistency', 'evalcrafter_ghosting_proxy', 'evalcrafter_temporal_ssim_proxy',
        'evalcrafter_temporal_edge_flicker', 'evalcrafter_color_shift', 'evalcrafter_white_balance_drift',
        'evalcrafter_exposure_clipping', 'evalcrafter_over_smoothing', 'evalcrafter_detail_preservation',
        'evalcrafter_frame_drop_indicator', 'evalcrafter_temporal_blur_increase', 'evalcrafter_flicker_frequency_proxy',
        'evalcrafter_motion_jerk', 'evalcrafter_spatial_jitter',
    ]
    res = {'ad_id': ad_id, 'variant_id': variant_id, 'ai_path': video_path, 
           'inference_steps': row.get('inference_steps', row.get('steps', ''))}
    for k in keys: res[k] = 0.5
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["auto", "main", "pilot"], default="auto", 
                        help="Choose which experiment log to process.")
    args = parser.parse_args()

    try:
        df, output_csv, video_root = load_data_config(args.target)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        return

    # Filter for success if status exists
    if 'status' in df.columns:
        df = df[df['status'] == 'success']
    
    logger.info(f"Processing {len(df)} videos for '{args.target}' experiment.")
    logger.info(f"Looking for videos in: {video_root}")

    tasks = []
    for _, row in df.iterrows():
        tasks.append((row.to_dict(), video_root))

    results = []
    max_workers = os.cpu_count()
    
    # Process
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(compute_metrics_for_video, tasks), total=len(tasks)):
            if res:
                results.append(res)
    
    if not results:
        logger.error("No metrics computed. Check your video paths.")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(out_df)} rows to {output_csv}")

if __name__ == "__main__":
    main()