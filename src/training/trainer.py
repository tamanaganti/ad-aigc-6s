"""Trainer for Ad-AIGC-6s: dual-task (retention + real score), early stopping, checkpointing."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles training, evaluation, and checkpointing.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(getattr(config, "device", "cuda") if torch.cuda.is_available() else "cpu")
        
        # Training Hyperparams - Safely cast to float/int
        self.epochs = int(getattr(config.training, "num_epochs", 50))
        self.batch_size = int(getattr(config.training, "batch_size", 128))
        self.patience = int(getattr(config.training, "patience", 10))
        self.clip_norm = float(getattr(config.training, "gradient_clip_norm", 1.0))
        
        # Paths
        self.output_dir = getattr(config.training, "output_dir", "experiments/final_run")
        
        # Explicit Logger Setup
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        log_path = os.path.join(self.output_dir, "training.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging training progress to: {log_path}")

        # Metrics history
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train(self, model, datasets, task_name="ad_aigc_6s_ai"):
        """Full training loop with Early Stopping."""
        model = model.to(self.device)
        
        # Optimizer
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        criterion = nn.MSELoss()

        # DataLoaders
        train_loader = DataLoader(
            datasets['train'], batch_size=self.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            datasets['validation'], batch_size=self.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )

        logger.info(f"Training on {len(datasets['train'])} samples")
        logger.info(f"Distributed: {torch.cuda.device_count() > 1} (World size: {torch.cuda.device_count()})")

        # Handle DDP if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for epoch in range(1, self.epochs + 1):
            # --- TRAIN ---
            train_loss, train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # --- VALIDATE ---
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Update LR
            if scheduler:
                scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} (Retained MAE: {train_metrics['retained_mae']:.4f}) | "
                f"Val Loss: {val_loss:.4f} (Retained MAE: {val_metrics['retained_mae']:.4f})"
            )

            # Checkpoint & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_model(model, "best_model.pt")
                logger.info(f"New best model (Val Loss: {val_loss:.4f})")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
        
        # Save final model (DDP Safe)
        self.save_model(model, "final_model.pt")
        
        # Return best model for immediate evaluation
        return self.load_model(model, os.path.join(self.output_dir, "best_model.pt"))

    def _train_epoch(self, model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0
        
        # Track metrics for BOTH tasks
        all_preds_retained, all_targets_retained = [], []
        all_preds_real, all_targets_real = [], []

        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in loop:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch) 
            # Output is dict: {'ad_aigc_6s_ai': tensor, 'ad_aigc_6s_real': tensor}

            # --- DUAL TASK LOSS ---
            # 1. Retention Loss (Main Task)
            pred_retained = outputs['ad_aigc_6s_ai'].squeeze()
            target_retained = batch['p_retained'].float()
            loss_retained = criterion(pred_retained, target_retained)

            # 2. Real Score Loss (Auxiliary Task)
            pred_real = outputs['ad_aigc_6s_real'].squeeze()
            target_real = batch['p_real'].float()
            loss_real = criterion(pred_real, target_real)

            # Total Loss (Equal weighting for now, can be tuned in config if needed)
            loss = loss_retained + loss_real
            
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Collect for metrics
            all_preds_retained.append(pred_retained.detach().cpu())
            all_targets_retained.append(target_retained.detach().cpu())
            all_preds_real.append(pred_real.detach().cpu())
            all_targets_real.append(target_real.detach().cpu())

        # Compute Epoch Metrics
        metrics = self._compute_metrics(
            torch.cat(all_preds_retained), torch.cat(all_targets_retained),
            torch.cat(all_preds_real), torch.cat(all_targets_real)
        )
        return total_loss / len(loader), metrics

    def _validate_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        
        all_preds_retained, all_targets_retained = [], []
        all_preds_real, all_targets_real = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                
                # Dual Loss Calculation
                pred_retained = outputs['ad_aigc_6s_ai'].squeeze()
                target_retained = batch['p_retained'].float()
                loss_retained = criterion(pred_retained, target_retained)

                pred_real = outputs['ad_aigc_6s_real'].squeeze()
                target_real = batch['p_real'].float()
                loss_real = criterion(pred_real, target_real)

                loss = loss_retained + loss_real
                total_loss += loss.item()

                all_preds_retained.append(pred_retained.cpu())
                all_targets_retained.append(target_retained.cpu())
                all_preds_real.append(pred_real.cpu())
                all_targets_real.append(target_real.cpu())

        metrics = self._compute_metrics(
            torch.cat(all_preds_retained), torch.cat(all_targets_retained),
            torch.cat(all_preds_real), torch.cat(all_targets_real)
        )
        return total_loss / len(loader), metrics

    def evaluate(self, model, dataset, task_name="ad_aigc_6s_ai"):
        """Final Test Evaluation."""
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        criterion = nn.MSELoss()
        _, metrics = self._validate_epoch(model, loader, criterion)
        
        logger.info(f"Final Test Results (Retained): MAE={metrics['retained_mae']:.4f}, RMSE={metrics['retained_rmse']:.4f}")
        logger.info(f"Final Test Results (Source):   MAE={metrics['real_mae']:.4f}, RMSE={metrics['real_rmse']:.4f}")
        
        return metrics

    def _compute_metrics(self, pred_ret, target_ret, pred_real, target_real):
        """Helper to compute MAE/RMSE for both heads."""
        return {
            "retained_mae": mean_absolute_error(target_ret, pred_ret),
            "retained_rmse": np.sqrt(mean_squared_error(target_ret, pred_ret)),
            "real_mae": mean_absolute_error(target_real, pred_real),
            "real_rmse": np.sqrt(mean_squared_error(target_real, pred_real))
        }

    def save_model(self, model, filename):
        """DDP-Safe Model Saving."""
        save_path = os.path.join(self.output_dir, filename)
        
        # Unwrap DDP if necessary
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        torch.save(state_dict, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, model, path):
        """Load model weights (handles unwrapped loading)."""
        # Unwrap DDP if model is currently wrapped but we are loading raw weights
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(path, map_location=self.device))
        else:
            model.load_state_dict(torch.load(path, map_location=self.device))
        return model

    def _get_optimizer(self, model):
        # Ensure lr and weight_decay are floats
        lr = float(getattr(self.config.training, "learning_rate", 1e-4))
        wd = float(getattr(self.config.training, "weight_decay", 1e-5))
        
        return optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )

    def _get_scheduler(self, optimizer):
        scheduler_type = getattr(self.config.training, "scheduler", "cosine")
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return None