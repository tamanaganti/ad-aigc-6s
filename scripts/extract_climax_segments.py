#!/usr/bin/env python3
"""
Extract 6-second climax segments.
ROBUST VERSION: Never crashes on download failure. Retries missing files.
"""

import os
import sys
import subprocess
import time
import random
from pathlib import Path
import logging
import csv
from typing import Optional, Tuple

# Log Setup
log_file = Path("logs/extraction.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Config
RAW_DIR = Path("data/raw/hussain/videos")
CLIMAX_DIR = Path("data/raw/hussain_climax")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLIMAX_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ID_LIST = Path("data/raw/hussain/final_video_id_list.csv")
DEFAULT_OUT_CSV = Path("data/metadata/hussain_climax.csv")


def download_video_youtube(video_id: str) -> Tuple[Optional[str], str]:
    """Enhanced download with better rate limiting and anti-ban measures."""
    output_path = RAW_DIR / f"{video_id}.mp4"

    if output_path.exists() and output_path.stat().st_size > 1000:
        return str(output_path), "exists"

    # Enhanced command with more anti-ban measures
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Lower quality to reduce load
        '-o', str(output_path),
        f'https://www.youtube.com/watch?v={video_id}',
        '--no-playlist', '--quiet', '--no-warnings',
        '--no-overwrites', '--force-ipv4',
        '--extractor-args', 'youtube:player_client=android_music',
        '--socket-timeout', '30',
        '--retries', '3',
        '--retry-sleep', '5',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '--referer', 'https://www.youtube.com/'
    ]

    try:
        # 5 Retries per video with exponential backoff
        for attempt in range(5):
            logger.info(f"   Attempt {attempt + 1}/5 for {video_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if output_path.exists() and output_path.stat().st_size > 1000:
                logger.info(f"   Downloaded {video_id}")
                return str(output_path), "downloaded"

            # Handle different error types
            stderr = result.stderr.lower()
            if "too many requests" in stderr or "429" in stderr:
                sleep_time = min(300, (2 ** attempt) * 60)  # Exponential backoff, max 5min
                logger.warning(f"   Rate limit hit. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            elif "unavailable" in stderr or "not available" in stderr:
                logger.info(f"   Video {video_id} unavailable, skipping")
                return None, "unavailable"
            elif "private" in stderr:
                logger.info(f"   Video {video_id} is private, skipping")
                return None, "private"
            elif attempt < 4:  # Don't sleep on last attempt
                sleep_time = (attempt + 1) * 10
                logger.info(f"   Download failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        logger.error(f"   Failed to download {video_id} after 5 attempts")
        return None, "failed"

    except subprocess.TimeoutExpired:
        logger.error(f"   Timeout downloading {video_id}")
        return None, "timeout"
    except Exception as e:
        logger.error(f"   Error downloading {video_id}: {e}")
        return None, f"error_{str(e)}"


def _ffprobe_duration_seconds(video_path: Path) -> Optional[float]:
    try:
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        return float(subprocess.check_output(dur_cmd).decode().strip())
    except Exception:
        return None


def extract_segment(raw_path: Path, video_id: str) -> bool:
    """Extract 6s segment using ffmpeg directly (faster/safer than moviepy)."""
    final_path = CLIMAX_DIR / f"{video_id}_real_6s.mp4"
    
    if final_path.exists() and final_path.stat().st_size > 1000:
        return True

    # Get duration
    duration = _ffprobe_duration_seconds(raw_path)
    if duration is None:
        return False

    # Calculate center
    start = max(0, (duration - 6.0) / 2.0)
    
    # Extract
    cmd = [
        'ffmpeg', '-y', '-ss', str(start), '-i', str(raw_path),
        '-t', '6.0', '-c:v', 'libx264', '-c:a', 'aac',
        '-strict', 'experimental', '-loglevel', 'error',
        str(final_path)
    ]
    subprocess.run(cmd)
    
    return final_path.exists()

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--id_list", type=str, default=str(DEFAULT_ID_LIST), help="CSV/text file containing one YouTube video id per line.")
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV), help="Where to write extraction metadata.")
    ap.add_argument("--download_youtube", action="store_true", help="If set, download missing raw videos from YouTube using yt-dlp.")
    ap.add_argument("--max_videos", type=int, default=0, help="Optional cap on number of ids processed (0 = no cap).")
    ap.add_argument("--raw_dir", type=str, default=str(RAW_DIR), help="Directory for temporary raw downloads / local raw videos.")
    ap.add_argument("--climax_dir", type=str, default=str(CLIMAX_DIR), help="Directory to write extracted 6s clips.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    climax_dir = Path(args.climax_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    climax_dir.mkdir(parents=True, exist_ok=True)

    # Load IDs (one per line). If not provided, fall back to annotation JSON keys.
    video_ids = []
    id_file = Path(args.id_list)
    if id_file.exists():
        with open(id_file, "r") as f:
            video_ids = [line.strip().replace("'", "").replace('"', "") for line in f if line.strip()]
        logger.info(f"Loaded {len(video_ids)} ids from: {id_file}")
    else:
        # Fallback: derive IDs from effective-score annotations (common Hussain layout)
        eff_json = Path("data/raw/hussain/video/cleaned_result/video_Effective_clean.json")
        if eff_json.exists():
            try:
                import json
                data = json.loads(eff_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    video_ids = list(map(str, data.keys()))
                    logger.warning(f"ID list not found at {id_file}; falling back to {len(video_ids)} ids from {eff_json}")
            except Exception:
                video_ids = []

    if not video_ids:
        logger.error(f"No video ids found. Provide --id_list or ensure {Path('data/raw/hussain/video/cleaned_result/video_Effective_clean.json')} exists.")
        return

    if args.max_videos and args.max_videos > 0:
        video_ids = video_ids[: int(args.max_videos)]
    
    logger.info(f"Processing {len(video_ids)} videos...")
    logger.info(f"   Output directory: {climax_dir}")
    logger.info(f"   Download from YouTube: {args.download_youtube}")

    success = 0
    failed = 0
    unavailable = 0
    existing = 0
    rows = []

    # Progress tracking
    last_progress_time = time.time()
    progress_interval = 50  # Log progress every 50 videos

    for idx, vid in enumerate(video_ids):
        try:
            final_path = climax_dir / f"{vid}_real_6s.mp4"

            # 1. Check if final exists
            if final_path.exists() and final_path.stat().st_size > 1000:
                existing += 1
                rows.append({"ad_id": vid, "real_path": str(final_path), "status": "exists"})
                if idx % progress_interval == 0:
                    elapsed = time.time() - last_progress_time
                    rate = progress_interval / elapsed if elapsed > 0 else 0
                    logger.info(f"[{idx}/{len(video_ids)}] {vid} exists (skipping). Rate: {rate:.1f} videos/sec")
                    last_progress_time = time.time()
                continue

            # 2. Locate raw input (local file first)
            raw_path = raw_dir / f"{vid}.mp4"
            status = "local"

            # 3. Optional download if raw does not exist
            if not raw_path.exists() or raw_path.stat().st_size <= 1000:
                if not args.download_youtube:
                    failed += 1
                    rows.append({"ad_id": vid, "real_path": "", "status": "missing_raw"})
                    logger.warning(f"[{idx}/{len(video_ids)}] Missing raw video for {vid}. Re-run with --download_youtube or provide local raw files.")
                    continue

                logger.info(f"[{idx}/{len(video_ids)}] Downloading {vid} from YouTube...")
                downloaded, status = download_video_youtube(vid)
                if not downloaded:
                    if status in ["unavailable", "private"]:
                        unavailable += 1
                        rows.append({"ad_id": vid, "real_path": "", "status": status})
                        logger.info(f"[{idx}/{len(video_ids)}] {vid} {status}, skipping")
                    else:
                        failed += 1
                        rows.append({"ad_id": vid, "real_path": "", "status": f"download_{status}"})
                        logger.warning(f"[{idx}/{len(video_ids)}] Download failed: {vid} ({status})")
                    continue
                raw_path = Path(downloaded)

            # 4. Extract segment
            logger.info(f"[{idx}/{len(video_ids)}] Extracting segment from {vid}...")
            if extract_segment(raw_path, vid):
                logger.info(f"[{idx}/{len(video_ids)}] Successfully processed {vid}")
                success += 1
                rows.append({"ad_id": vid, "real_path": str(final_path), "status": "extracted"})
            else:
                logger.warning(f"[{idx}/{len(video_ids)}] Extraction failed {vid}")
                failed += 1
                rows.append({"ad_id": vid, "real_path": "", "status": "extract_failed"})

            # 5. Cleanup raw downloads (only if we downloaded)
            if args.download_youtube and status in ["downloaded", "exists"]:
                try:
                    if raw_path.exists():
                        raw_path.unlink()
                except Exception as e:
                    logger.warning(f"   Failed to cleanup {raw_path}: {e}")

            # 6. Rate limiting - add small delay between videos
            if idx % 10 == 0 and idx > 0:
                logger.info("   Pausing 5s between batches.")
                time.sleep(5)

        except KeyboardInterrupt:
            logger.info(f"Interrupted at video {idx}/{len(video_ids)}. Progress saved.")
            break
        except Exception as e:
            logger.error(f"[{idx}/{len(video_ids)}] CRITICAL ERROR on {vid}: {e}")
            failed += 1
            rows.append({"ad_id": vid, "real_path": "", "status": f"error_{type(e).__name__}"})
            continue # NEVER CRASH

    # Write metadata CSV (helps downstream scripts identify which clips exist)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ad_id", "real_path", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Wrote extraction metadata to {out_csv}")

    # Final summary with detailed breakdown
    logger.info("="*60)
    logger.info("DOWNLOAD EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info("FINAL STATISTICS:")
    logger.info(f"   Total Videos Processed: {len(video_ids)}")
    logger.info(f"   Successful Extractions: {success}")
    logger.info(f"   Already Existed: {existing}")
    logger.info(f"   Failed Downloads: {failed}")
    logger.info(f"   Unavailable/Private: {unavailable}")
    logger.info(f"   Output Directory: {climax_dir}")
    logger.info(f"   Valid Clips Ready: {success + existing}/{len(video_ids)} ({(success + existing)/len(video_ids)*100:.1f}%)")
    logger.info("="*60)

if __name__ == "__main__":
    main()