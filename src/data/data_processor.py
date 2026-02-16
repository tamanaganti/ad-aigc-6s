"""
Data processor for Ad-AIGC-6s.
UPDATED: Fixes config path access and uses local CLIP to avoid hangs.
"""

import logging
import os
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, Features, Value, Sequence

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.clip_processor = None

        # Resolve config: data.ad_aigc_6s or root ad_aigc_6s
        if getattr(config, "data", None) and getattr(config.data, "ad_aigc_6s", None):
            self.data_cfg = config.data.ad_aigc_6s
        elif getattr(config, "ad_aigc_6s", None):
            self.data_cfg = config.ad_aigc_6s
        else:
            raise ValueError("Config missing 'ad_aigc_6s' section (checked 'config.data.ad_aigc_6s' and 'config.ad_aigc_6s')")

        base_path = Path(self.data_cfg.path)
        self.cache_path = base_path.parent / f"{base_path.stem}_cached.csv"

    # ---------------------------
    # public
    # ---------------------------
    def load_dataset_from_csv(self) -> Dict[str, Dataset]:
        df = self._load_or_build_cache()

        # always build/overwrite quality_vector from current columns
        df = self._build_quality_vector(df)

        # compute q_ai if missing
        if "q_ai" not in df.columns:
            df = self._compute_aggregated_quality(df)

        # hard enforce dtypes + fixed lengths BEFORE split and HF conversion
        df = self._enforce_and_clean(df)

        train_df, val_df, test_df = self._split(df)

        logger.info(
            "Splits: Train=%d, Val=%d, Test=%d",
            len(train_df), len(val_df), len(test_df)
        )

        return {
            "train": self._to_hf_dataset(train_df),
            "validation": self._to_hf_dataset(val_df),
            "test": self._to_hf_dataset(test_df),
        }

    # ---------------------------
    # cache / load
    # ---------------------------
    def _load_or_build_cache(self) -> pd.DataFrame:
        # 1) try cache
        if os.path.exists(self.cache_path):
            logger.info("Found cache: %s", str(self.cache_path))
            try:
                df = pd.read_csv(self.cache_path)

                if "visual_embedding" in df.columns:
                    # parse embedding strings -> list
                    if isinstance(df["visual_embedding"].iloc[0], str):
                        df["visual_embedding"] = df["visual_embedding"].apply(self._safe_literal_eval)

                    # quick length check
                    v0 = df["visual_embedding"].iloc[0]
                    if not self._is_vec(v0, 512):
                        logger.warning("Cache embedding length mismatch. Recomputing cache.")
                        raise ValueError("cache embedding length mismatch")

                    return df

                logger.warning("Cache missing visual_embedding column. Recomputing cache.")
            except Exception as e:
                logger.warning("Cache load failed (%s). Recomputing cache.", str(e))

        # 2) build from raw
        csv_path = self.data_cfg.path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        logger.info("Loading raw dataset: %s", csv_path)
        df = pd.read_csv(csv_path)

        if "ai_path" not in df.columns and "ai_video_path" in df.columns:
            df["ai_path"] = df["ai_video_path"]

        logger.info("Generating visual features (CLIP, middle frame)...")
        df = self._cache_visual_features(df)

        # save cache (store embeddings as JSON-ish string)
        logger.info("Saving cache: %s", str(self.cache_path))
        df_to_save = df.copy()
        df_to_save["visual_embedding"] = df_to_save["visual_embedding"].apply(
            lambda x: x if isinstance(x, str) else str(list(map(float, x)))
        )
        df_to_save.to_csv(self.cache_path, index=False)

        return df

    # ---------------------------
    # enforce types/shapes
    # ---------------------------
    def _enforce_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["ad_id", "p_real", "p_retained", "s_sim", "s_drift", "visual_embedding", "quality_vector"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # make sure embeddings are parsed
        if isinstance(df["visual_embedding"].iloc[0], str):
            df["visual_embedding"] = df["visual_embedding"].apply(self._safe_literal_eval)

        if isinstance(df["quality_vector"].iloc[0], str):
            df["quality_vector"] = df["quality_vector"].apply(self._safe_literal_eval)

        # enforce vector lengths + float32
        def fix_vec(x, n, fill=0.0):
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if not isinstance(x, list):
                x = [fill] * n
            # trim/pad
            if len(x) < n:
                x = x + [fill] * (n - len(x))
            elif len(x) > n:
                x = x[:n]
            # float
            out = []
            for v in x:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float(fill))
            return out

        df["visual_embedding"] = df["visual_embedding"].apply(lambda x: fix_vec(x, 512, 0.0))
        df["quality_vector"] = df["quality_vector"].apply(lambda x: fix_vec(x, 33, 0.5))

        # numeric cols float32
        for col in ["p_real", "p_retained", "s_sim", "s_drift"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

        if "q_ai" in df.columns:
            df["q_ai"] = pd.to_numeric(df["q_ai"], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            df["q_ai"] = np.float32(0.0)

        # final integrity check (fail fast)
        v0 = df["visual_embedding"].iloc[0]
        q0 = df["quality_vector"].iloc[0]
        if not self._is_vec(v0, 512):
            raise ValueError(f"visual_embedding bad shape after clean: len={len(v0) if hasattr(v0,'__len__') else 'NA'}")
        if not self._is_vec(q0, 33):
            raise ValueError(f"quality_vector bad shape after clean: len={len(q0) if hasattr(q0,'__len__') else 'NA'}")

        logger.info("Data integrity check passed: visual_embedding[0]=512, quality_vector[0]=33")
        return df

    def _is_vec(self, x, n: int) -> bool:
        return isinstance(x, (list, np.ndarray)) and len(x) == n

    def _safe_literal_eval(self, x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x

    # ---------------------------
    # split
    # ---------------------------
    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Use the data_cfg we found in init
        train_ratio = float(getattr(self.data_cfg, "train_split", 0.8))
        val_ratio = float(getattr(self.data_cfg, "val_split", 0.1))
        test_ratio = float(getattr(self.data_cfg, "test_split", 1.0 - train_ratio - val_ratio))

        # normalize if user provided only train/val
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            train_ratio = train_ratio / total
            val_ratio = val_ratio / total
            test_ratio = test_ratio / total

        train_df, rest_df = train_test_split(df, test_size=(1.0 - train_ratio), random_state=42, shuffle=True)
        # split rest into val/test proportionally
        if len(rest_df) == 0:
            raise ValueError("Split resulted in empty val/test. Check split ratios.")
        val_frac_of_rest = val_ratio / (val_ratio + test_ratio + 1e-12)
        val_df, test_df = train_test_split(rest_df, test_size=(1.0 - val_frac_of_rest), random_state=42, shuffle=True)
        return train_df, val_df, test_df

    # ---------------------------
    # HF dataset conversion
    # ---------------------------
    def _to_hf_dataset(self, df: pd.DataFrame) -> Dataset:
        # keep only columns needed by the model/trainer and for analysis
        keep_cols = ["ad_id", "p_real", "p_retained", "q_ai", "s_sim", "s_drift", "visual_embedding", "quality_vector"]

        # include raw metric columns too if present (helps analysis)
        metric_cols = [c for c in df.columns if c.startswith("vbench_") or c.startswith("evalcrafter_")]
        keep_cols = keep_cols + [c for c in metric_cols if c not in keep_cols]

        df_clean = df[keep_cols].copy()

        # Features: use fixed-length Sequence (stable across datasets versions)
        features = Features({
            "ad_id": Value("string"),
            "p_real": Value("float32"),
            "p_retained": Value("float32"),
            "q_ai": Value("float32"),
            "s_sim": Value("float32"),
            "s_drift": Value("float32"),
            "visual_embedding": Sequence(Value("float32"), length=512),
            "quality_vector": Sequence(Value("float32"), length=33),
        })

        # add metric cols as float32
        for c in metric_cols:
            if c not in features:
                features[c] = Value("float32")
                df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0).astype(np.float32)

        ds = Dataset.from_pandas(df_clean.reset_index(drop=True), features=features)

        # torch tensors
        tensor_cols = ["p_real", "p_retained", "q_ai", "s_sim", "s_drift", "visual_embedding", "quality_vector"]
        ds.set_format(type="torch", columns=tensor_cols)
        return ds

    # ---------------------------
    # quality vector + q_ai
    # ---------------------------
    def _build_quality_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        vbench = sorted([c for c in df.columns if c.startswith("vbench_")])
        evalcrafter = sorted([c for c in df.columns if c.startswith("evalcrafter_")])
        quality_cols = vbench + evalcrafter

        if len(quality_cols) != 33:
            logger.warning("Expected 33 metric columns, found %d. Will still build quality_vector with padding/truncation.", len(quality_cols))

        if quality_cols:
            q = df[quality_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).astype(np.float32).values.tolist()
        else:
            q = [[0.5] * 33 for _ in range(len(df))]

        df = df.copy()
        df["quality_vector"] = q
        return df

    def _compute_aggregated_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        vbench = [c for c in df.columns if c.startswith("vbench_")]
        evalcrafter = [c for c in df.columns if c.startswith("evalcrafter_")]

        df = df.copy()
        if vbench and evalcrafter:
            df["q_ai"] = (0.6 * df[vbench].mean(axis=1) + 0.4 * df[evalcrafter].mean(axis=1)).astype(np.float32)
        elif vbench:
            df["q_ai"] = df[vbench].mean(axis=1).astype(np.float32)
        elif evalcrafter:
            df["q_ai"] = df[evalcrafter].mean(axis=1).astype(np.float32)
        else:
            df["q_ai"] = np.float32(0.5)
        return df

    # ---------------------------
    # CLIP embedding (middle frame)
    # ---------------------------
    def _cache_visual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_clip()
        embeddings: List[List[float]] = []

        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Caching CLIP Features")
        except Exception:
            iterator = df.iterrows()

        for _, row in iterator:
            path = row.get("ai_path", row.get("ai_video_path"))
            embeddings.append(self._extract_clip_embedding(path))

        df = df.copy()
        df["visual_embedding"] = embeddings
        return df

    def _load_clip(self):
        import os
        from pathlib import Path
        import torch
        from transformers import CLIPModel, CLIPImageProcessor

        # whatever your code currently sets:
        local_path = Path("models/clip-vit-base-patch32")

        # prefer local if it has real weights
        has_weights = (
            (local_path / "model.safetensors").exists()
            or (local_path / "pytorch_model.bin").exists()
            or any(local_path.glob("*.safetensors"))
            or any(local_path.glob("pytorch_model*.bin"))
        )

        model_id = "openai/clip-vit-base-patch32"

        if has_weights:
            logger.info("Loading CLIP from local path: %s", local_path)
            self.clip_model = CLIPModel.from_pretrained(
                str(local_path),
                local_files_only=True,
            ).to(self.device)
            self.clip_processor = CLIPImageProcessor.from_pretrained(
                str(local_path),
                local_files_only=True,
                use_fast=False,
            )
        else:
            logger.info("Local CLIP weights not found at %s. Loading from HF model id: %s", local_path, model_id)
            # allow download (cached under HF_HOME / default cache)
            self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.clip_processor = CLIPImageProcessor.from_pretrained(model_id, use_fast=False)

        self.clip_model.eval()

    def _extract_clip_embedding(self, video_path: Optional[str]) -> List[float]:
        try:
            if not video_path or not os.path.exists(video_path):
                return [0.0] * 512

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return [0.0] * 512

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            mid = max(0, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return [0.0] * 512

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.clip_model.vision_model(**inputs)
                emb = self.clip_model.visual_projection(out.pooler_output)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

            vec = emb.detach().cpu().numpy().reshape(-1).astype(np.float32).tolist()
            if len(vec) != 512:
                return [0.0] * 512
            return vec
        except Exception:
            return [0.0] * 512