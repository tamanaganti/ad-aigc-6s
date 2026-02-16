"""
Multimodal feature extractor.
LIGHTWEIGHT VERSION: Optimized for Ad-AIGC-6s pipeline.
Removes heavy audio/biometric dependencies to prevent OOM on SVD videos.
"""

import logging
import os
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import CLIPModel, CLIPProcessor
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FeatureExtractor:
    """
    Lightweight feature extractor.
    Primarily used for CLIP visual embeddings. 
    Audio/Biometrics/VBench are disabled to rely on pre-computed CSV metrics.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_model = None
        self.clip_processor = None
        
        # Lazy loading flag
        self._models_loaded = False

    def _load_models(self):
        """Lazy load CLIP only when needed."""
        if self._models_loaded: return
        
        try:
            model_name = self.config.visual.get("model_name", "ViT-B/32")
            logger.info(f"Loading CLIP ({model_name}) for visual extraction...")
            
            # Handle different CLIP naming conventions
            hf_name = f"openai/clip-{model_name.lower()}"
            if "vit" in model_name.lower():
                hf_name = "openai/clip-vit-base-patch32" # Fallback to standard
                
            self.clip_model = CLIPModel.from_pretrained(hf_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(hf_name)
            self.clip_model.eval()
            self._models_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def extract_visual_embedding(self, video_path):
        """
        Extracts a single pooled visual embedding for a video.
        Used by DataProcessor for caching.
        """
        if not self._models_loaded:
            self._load_models()
            
        try:
            if not os.path.exists(video_path):
                return np.zeros(512)

            # Cap read - Middle Frame Strategy (Efficient)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return np.zeros(512)

            # Preprocess
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            # Inference
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()

        except Exception as e:
            logger.warning(f"Visual extraction failed for {video_path}: {e}")
            return np.zeros(512)

    def extract_features(self, datasets):
        """
        Legacy pass-through. 
        In this pipeline, features are pre-computed in CSV.
        """
        logger.info("Feature extraction skipped (Features loaded from CSV).")
        return datasets