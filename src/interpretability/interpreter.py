import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelInterpreter:
    """
    Analyzes the trained Quality-Gated Attention model (gating and feature importance).
    """

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs")) / "interpretability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Hardcoded 33 metrics in order (must match training exactly)
        self.quality_cols = [
            'vbench_motion_smoothness', 'vbench_temporal_flicker', 'vbench_subject_consistency', 
            'vbench_aesthetic_quality', 'vbench_dynamic_degree', 'vbench_brightness_consistency', 
            'vbench_contrast_consistency', 'vbench_color_consistency', 'vbench_saturation_consistency', 
            'vbench_sharpness_consistency', 'vbench_noise_level', 'vbench_edge_consistency', 
            'vbench_camera_shake', 'vbench_scene_change_rate', 'vbench_blockiness', 'vbench_black_frame_ratio',
            'evalcrafter_flow_magnitude_mean', 'evalcrafter_flow_magnitude_std', 'evalcrafter_flow_direction_std', 
            'evalcrafter_flow_inconsistency', 'evalcrafter_ghosting_proxy', 'evalcrafter_temporal_ssim_proxy', 
            'evalcrafter_temporal_edge_flicker', 'evalcrafter_color_shift', 'evalcrafter_white_balance_drift', 
            'evalcrafter_exposure_clipping', 'evalcrafter_over_smoothing', 'evalcrafter_detail_preservation', 
            'evalcrafter_frame_drop_indicator', 'evalcrafter_temporal_blur_increase', 
            'evalcrafter_flicker_frequency_proxy', 'evalcrafter_motion_jerk', 'evalcrafter_spatial_jitter'
        ]

    def analyze(self, model, test_dataset):
        logger.info("Starting interpretability analysis.")
        model = model.to(self.device)
        model.eval()
        
        # 1. Gating Analysis
        self._analyze_gating_mechanism(model, test_dataset)
        
        # 2. Feature Importance (Diagnostic Mode)
        self._analyze_feature_importance(model, test_dataset)

    def _analyze_gating_mechanism(self, model, dataset):
        logger.info("   Computing Gating Activations...")
        gate_values = []
        quality_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Extracting Gates"):
                item = dataset[i]
                # Use clone().detach() to avoid warnings
                q_vec = torch.tensor(item['quality_vector']).float().unsqueeze(0).to(self.device)
                try:
                    # Explicit Forward Path
                    q_emb = model.qual_proj(q_vec).unsqueeze(1) 
                    gate_out = model.fusion.gate_net(q_emb.squeeze(1)) 
                    alpha = gate_out.mean().item()
                    
                    gate_values.append(alpha)
                    quality_scores.append(float(item['q_ai']))
                except AttributeError:
                    continue

        if gate_values:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=quality_scores, y=gate_values, alpha=0.6, color='crimson')
            sns.regplot(x=quality_scores, y=gate_values, scatter=False, color='black', line_kws={"linestyle": "--"})
            plt.title("Gating Mechanism: Modulation of Visual Features")
            plt.savefig(self.output_dir / "figure3_gating_mechanism.png")
            plt.close()
            
            corr = np.corrcoef(quality_scores, gate_values)[0, 1]
            logger.info(f"   Saved Figure 3. Correlation (Quality vs Gate): {corr:.4f}")

    def _analyze_feature_importance(self, model, dataset):
        """Calculates permutation importance for the model."""
        logger.info("   Computing Feature Importance (Permutation Method)...")
        
        def get_preds(output):
            if isinstance(output, dict):
                # 1. PRIORITY: Look for the specific AI task head
                if 'ad_aigc_6s_ai' in output:
                    return output['ad_aigc_6s_ai']
                
                # 2. Fallbacks
                for key in ['retention_pred', 'pred', 'logits']:
                    if key in output:
                        return output[key]
                        
                keys = list(output.keys())
                logger.warning(f"Could not find 'ad_aigc_6s_ai'. Using '{keys[0]}'.")
                return output[keys[0]]
            return output

        # 1. Prepare Batch
        try:
            full_batch = self._collate_all(dataset)
        except Exception as e:
            logger.error(f"   Could not collate dataset: {e}")
            return

        # 2. Compute baseline
        with torch.no_grad():
            base_out = model(full_batch)
            base_tensor = get_preds(base_out)
            base_preds = base_tensor.cpu().numpy().flatten()
            base_labels = full_batch['retention_labels'].cpu().numpy().flatten()
            
        base_rmse = np.sqrt(np.mean((base_preds - base_labels) ** 2))
        
        if base_rmse > 0.25:
            logger.error("   Baseline RMSE too high; skipping permutation plot. Check labels and visual_embedding.")
            return

        # 3. Permutation Loop (Only runs if RMSE is sane)
        features_to_test = ['s_drift', 's_sim'] + self.quality_cols
        importances = {}
        
        for feat_name in tqdm(features_to_test, desc="Permuting Features"):
            corrupted_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in full_batch.items()}
            
            if feat_name == 's_drift':
                idx = torch.randperm(corrupted_batch['s_drift'].size(0))
                corrupted_batch['s_drift'] = corrupted_batch['s_drift'][idx]
            elif feat_name == 's_sim':
                idx = torch.randperm(corrupted_batch['s_sim'].size(0))
                corrupted_batch['s_sim'] = corrupted_batch['s_sim'][idx]
            elif feat_name in self.quality_cols:
                col_idx = self.quality_cols.index(feat_name)
                perm_idx = torch.randperm(corrupted_batch['quality_vector'].size(0))
                corrupted_batch['quality_vector'][:, col_idx] = corrupted_batch['quality_vector'][perm_idx, col_idx]
            
            with torch.no_grad():
                out = model(corrupted_batch)
                preds = get_preds(out).cpu().numpy().flatten()
            
            perm_rmse = np.sqrt(np.mean((preds - base_labels) ** 2))
            importances[feat_name] = perm_rmse - base_rmse

        # 4. Plot
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        keys = [x[0].replace('vbench_', '').replace('evalcrafter_', '') for x in sorted_feats]
        vals = [x[1] for x in sorted_feats]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=vals, y=keys, palette="viridis")
        plt.title("Top Factors Influencing Ad Retention")
        plt.tight_layout()
        plt.savefig(self.output_dir / "figure2_feature_importance.png")
        plt.close()
        logger.info("   Saved Figure 2 (Feature Importance)")

    def _collate_all(self, dataset):
        """Stacks dataset into one GPU batch."""
        batch = {
            'quality_vector': [], 's_sim': [], 's_drift': [], 
            'visual_embedding': [], 'retention_labels': []
        }
        
        for i in range(len(dataset)):
            item = dataset[i]
            # Use safe tensor conversion
            batch['quality_vector'].append(torch.tensor(item['quality_vector']).float())
            batch['s_sim'].append(torch.tensor(item['s_sim']).float())
            batch['s_drift'].append(torch.tensor(item['s_drift']).float())
            batch['retention_labels'].append(torch.tensor(item['p_retained']).float())
            
            if 'visual_embedding' in item:
                 batch['visual_embedding'].append(torch.tensor(item['visual_embedding']).float())
            else:
                 batch['visual_embedding'].append(torch.zeros(512))

        final_batch = {k: torch.stack(v).to(self.device) for k, v in batch.items()}
        return final_batch