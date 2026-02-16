#!/usr/bin/env python3
"""
Fixes the Hussain Annotations.
1. Deletes the corrupted folder.
2. Re-downloads the zip.
3. Extracts ONLY the JSON files (ignoring directory structure issues).
"""

import os
import zipfile
import shutil
from pathlib import Path
import tempfile
import urllib.request

def fix_annotations():
    base_dir = Path("data/raw/hussain")
    zip_path = base_dir / "annotations_videos.zip"
    extract_dir = base_dir / "annotations_videos"
    cleaned_dir = base_dir / "video" / "cleaned_result"
    
    print(f"Fixing annotations in {base_dir}...")
    
    # 1. Clean Slate
    if extract_dir.exists():
        print("   Removing old annotations folder...")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Force Re-Download (The old zip might be partial)
    url = "https://people.cs.pitt.edu/~kovashka/ads/annotations_videos.zip"
    print(f"   Downloading metadata from {url}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(zip_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # 3. Robust extraction: flatten into annotations_videos/ and also place key files
    print("   Extracting JSON files...")
    flat_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if not member.filename.endswith(".json") or member.filename.startswith("__MACOSX"):
                    continue

                filename = Path(member.filename).name
                target_path = extract_dir / filename

                with z.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                flat_count += 1

        print(f"Extracted {flat_count} JSON files into {extract_dir}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip.")
        return

    # 4. Ensure the expected path exists (scripts reference video_Effective_clean.json under cleaned_result)
    preferred = extract_dir / "video_Effective_clean.json"
    if preferred.exists():
        shutil.copy2(preferred, cleaned_dir / preferred.name)
        print(f"Copied {preferred.name} to {cleaned_dir}")
    else:
        # fallback: copy any 'Effective' jsons if present
        candidates = list(extract_dir.glob("*Effective*.json"))
        for c in candidates:
            shutil.copy2(c, cleaned_dir / c.name)
        if candidates:
            print(f"video_Effective_clean.json not found; copied {len(candidates)} Effective*.json files to {cleaned_dir}")
        else:
            print("No Effective*.json files found in extracted annotations; downstream computations may fail until annotations are provided.")

if __name__ == "__main__":
    fix_annotations()