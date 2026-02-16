# scripts/compute_similarity_and_labels.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

K_FRAMES_DEFAULT = 8


def load_effective_scores(json_path: Path) -> Dict[str, float]:
    if not json_path.exists():
        raise FileNotFoundError(f"Missing Hussain annotations JSON: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON mapping of video_id -> effective_score")
    return {str(k): float(v) for k, v in data.items()}


def scale_p_real(effective_score: float) -> float:
    return (float(effective_score) - 1.0) / 4.0


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ad_id" not in out.columns:
        for alt in ["video_id", "id"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "ad_id"})
                break
    if "ad_id" not in out.columns:
        raise ValueError(f"Missing ad_id column. Found: {list(df.columns)}")
    out["ad_id"] = out["ad_id"].astype(str)
    if "variant_id" in out.columns:
        out["variant_id"] = out["variant_id"].astype(str)
    return out


def _load_generation_log(gen_log_csv: Path) -> pd.DataFrame:
    if not gen_log_csv.exists():
        raise FileNotFoundError(f"Generation log CSV not found: {gen_log_csv}")

    gen = pd.read_csv(gen_log_csv)
    gen = _normalize_ids(gen)

    # inference_steps from inference_steps or quality_level or alternates
    if "inference_steps" in gen.columns:
        gen["inference_steps"] = pd.to_numeric(gen["inference_steps"], errors="coerce")
    elif "quality_level" in gen.columns:
        gen["inference_steps"] = pd.to_numeric(gen["quality_level"], errors="coerce")
    else:
        for alt in ["steps", "num_inference_steps"]:
            if alt in gen.columns:
                gen = gen.rename(columns={alt: "inference_steps"})
                gen["inference_steps"] = pd.to_numeric(gen["inference_steps"], errors="coerce")
                break

    # parse from variant_id if still missing
    if "inference_steps" not in gen.columns or gen["inference_steps"].isna().all():
        if "variant_id" not in gen.columns:
            raise ValueError(
                "Generation log must include inference_steps (or quality_level/steps/num_inference_steps) "
                "or include variant_id to parse steps."
            )
        import re

        def _parse_steps(v):
            if pd.isna(v):
                return np.nan
            s = str(v).lower()
            m = re.search(r"(?:^|[_-])q(\d+)(?:[_-]|$)", s)  # q10
            if m:
                return float(m.group(1))
            m = re.search(r"(?:steps|step)[=_-]?(\d+)", s)  # steps10
            if m:
                return float(m.group(1))
            m = re.search(r"(?:^|[_-])s(\d+)(?:[_-]|$)", s)  # _s10_
            if m:
                return float(m.group(1))
            return np.nan

        gen["inference_steps"] = gen["variant_id"].apply(_parse_steps)

    return gen


def attach_generation_fields(df: pd.DataFrame, gen_log_csv: Path, fields: List[str]) -> pd.DataFrame:
    gen = _load_generation_log(gen_log_csv)
    out = _normalize_ids(df)

    missing = [f for f in fields if f not in gen.columns]
    if missing:
        raise ValueError(f"Generation log missing fields {missing}. Found: {list(gen.columns)}")

    # merge with suffixes and coalesce into the base columns
    if "variant_id" in out.columns and "variant_id" in gen.columns:
        merged = out.merge(
            gen[["ad_id", "variant_id"] + fields],
            on=["ad_id", "variant_id"],
            how="left",
            suffixes=("", "_gen"),
        )
    else:
        merged = out.merge(
            gen[["ad_id"] + fields].drop_duplicates("ad_id"),
            on="ad_id",
            how="left",
            suffixes=("", "_gen"),
        )

    for f in fields:
        f_gen = f"{f}_gen"
        if f_gen in merged.columns:
            if f in merged.columns:
                merged[f] = merged[f].where(~merged[f].isna(), merged[f_gen])
            else:
                merged[f] = merged[f_gen]
            merged = merged.drop(columns=[f_gen])

    return merged


def add_q_ai_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    vbench_cols = [c for c in df.columns if c.startswith("vbench_")]
    eval_cols = [c for c in df.columns if c.startswith("evalcrafter_")]
    if not vbench_cols or not eval_cols:
        raise ValueError(
            f"Expected vbench_* and evalcrafter_* columns. Found vbench={len(vbench_cols)}, eval={len(eval_cols)}"
        )
    out = df.copy()
    out["q_ai_aggregate"] = 0.6 * out[vbench_cols].mean(axis=1) + 0.4 * out[eval_cols].mean(axis=1)
    return out


def uniform_frame_indices(n_frames: int, k: int) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if k <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, n_frames - 1, num=k)
    return np.unique(idx.astype(int))


def _read_frames_cv2(video_path: str, indices: np.ndarray):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []
    for i in indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    if not frames:
        raise ValueError(f"Failed to read any frames from {video_path} at indices {indices.tolist()}")
    return frames


def build_clip_embedder(device: str = "cpu"):
    import torch
    from transformers import CLIPModel, CLIPImageProcessor

    name = "openai/clip-vit-base-patch32"
    image_processor = CLIPImageProcessor.from_pretrained(name, use_fast=False)
    model = CLIPModel.from_pretrained(name)
    model.eval().to(device)

    @torch.no_grad()
    def embed_images(pil_images):
        inputs = image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # vision forward -> pooled -> projection (always tensors)
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # (B, hidden)
        feats = model.visual_projection(pooled)  # (B, D)

        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().numpy()

    return embed_images


def mean_video_embedding(video_path: str, k_frames: int, embed_images_fn, sampling: str = "uniform") -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n <= 0:
        raise ValueError(f"Video has 0 frames or frame count unavailable: {video_path}")

    if sampling != "uniform":
        raise ValueError(f"Unsupported sampling: {sampling}")

    idx = uniform_frame_indices(n, k_frames)
    frames = _read_frames_cv2(video_path, idx)

    feats = embed_images_fn(frames)
    emb = feats.mean(axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality_csv", required=True)
    ap.add_argument("--annotations_json", required=True)
    ap.add_argument("--generation_log_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--lambda_val", type=float, default=0.5)
    ap.add_argument("--k_frames", type=int, default=K_FRAMES_DEFAULT)
    ap.add_argument("--frame_sampling", type=str, default="uniform", choices=["uniform"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--subset_n", type=int, default=0)
    ap.add_argument("--subset_seed", type=int, default=42)
    args = ap.parse_args()

    lam = float(args.lambda_val)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda_val must be in [0,1]")
    if args.k_frames <= 0:
        raise ValueError("k_frames must be >= 1")

    quality_df = pd.read_csv(args.quality_csv)
    quality_df = _normalize_ids(quality_df)

    # attach real_path, ai_path, inference_steps without requiring CSV edits
    quality_df = attach_generation_fields(
        quality_df,
        Path(args.generation_log_csv),
        fields=["real_path", "ai_path", "inference_steps"],
    )

    if "real_path" not in quality_df.columns or quality_df["real_path"].isna().all():
        raise ValueError("real_path missing after merge. Check generation log has real_path.")
    if "ai_path" not in quality_df.columns or quality_df["ai_path"].isna().all():
        raise ValueError("ai_path missing after merge. Check generation log has ai_path.")

    quality_df = add_q_ai_aggregate(quality_df)

    scores = load_effective_scores(Path(args.annotations_json))
    missing_ids = [vid for vid in quality_df["ad_id"].tolist() if vid not in scores]
    if missing_ids:
        raise KeyError(f"{len(missing_ids)} ad_id missing in annotations JSON. Sample: {missing_ids[:10]}")

    quality_df["p_real"] = quality_df["ad_id"].map(lambda vid: scale_p_real(scores[str(vid)])).astype(float)

    # deterministic subset option
    if args.subset_n and args.subset_n > 0:
        rng = np.random.default_rng(int(args.subset_seed))
        take = min(int(args.subset_n), len(quality_df))
        chosen = rng.choice(np.arange(len(quality_df)), size=take, replace=False)
        quality_df = quality_df.iloc[np.sort(chosen)].reset_index(drop=True)

    embed_images_fn = build_clip_embedder(device=args.device)

    s_sim_vals: List[float] = []
    errors: List[Tuple[str, str, str]] = []

    for _, row in quality_df.iterrows():
        ad_id = str(row["ad_id"])
        var_id = str(row["variant_id"]) if "variant_id" in quality_df.columns else ad_id
        real_path = str(row["real_path"])
        ai_path = str(row["ai_path"])
        try:
            real_emb = mean_video_embedding(real_path, args.k_frames, embed_images_fn, sampling=args.frame_sampling)
            ai_emb = mean_video_embedding(ai_path, args.k_frames, embed_images_fn, sampling=args.frame_sampling)
            s_sim_vals.append(cosine_sim(real_emb, ai_emb))
        except Exception as e:
            errors.append((ad_id, var_id, str(e)))
            s_sim_vals.append(np.nan)

    quality_df["s_sim"] = pd.to_numeric(pd.Series(s_sim_vals), errors="coerce")

    if quality_df["s_sim"].isna().any():
        bad = int(quality_df["s_sim"].isna().sum())
        raise RuntimeError(f"Failed s_sim for {bad} rows. Sample: {errors[:10]}")

    quality_df["s_drift"] = 1.0 - quality_df["s_sim"].astype(float)
    quality_df["lambda"] = lam
    quality_df["k_frames"] = int(args.k_frames)
    quality_df["frame_sampling"] = args.frame_sampling

    quality_df["retention_factor"] = lam * quality_df["q_ai_aggregate"].astype(float) + (1.0 - lam) * quality_df["s_sim"].astype(float)
    quality_df["p_retained"] = quality_df["p_real"].astype(float) * quality_df["retention_factor"].astype(float)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(args.out_csv, index=False)
    print(f"Wrote labeled dataset: {args.out_csv} (rows={len(quality_df)})")


if __name__ == "__main__":
    main()
