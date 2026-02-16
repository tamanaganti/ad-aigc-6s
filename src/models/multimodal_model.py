import torch
import torch.nn as nn

class QualityGatedAttention(nn.Module):
    """
    Modulates content features based on quality confidence.
    content_emb: [B, 1, H]
    quality_emb:  [B, 1, H]
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, content_emb, quality_emb):
        # Gate: Quality -> Weight [0..1]
        gate = self.gate_net(quality_emb.squeeze(1)).unsqueeze(1)  # [B, 1, H]
        gated_content = content_emb * gate

        # Attention: Gated Content attends to Quality
        attn_out, _ = self.mha(
            query=gated_content,
            key=quality_emb,
            value=quality_emb
        )
        return self.norm(content_emb + self.dropout(attn_out))


class MultimodalAdModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- 1. CONFIG PARSING ---
        self.fusion_type = config.fusion.type
        hidden_dim = config.fusion.hidden_dim
        num_heads = config.fusion.num_heads
        dropout = config.fusion.dropout

        dim_visual = int(config.input_dims.visual)
        dim_quality = int(config.input_dims.quality)
        dim_drift = int(config.input_dims.semantic_drift)

        # --- 2. INPUT PROJECTIONS ---
        self.vis_proj = nn.Sequential(
            nn.Linear(dim_visual, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.qual_proj = nn.Sequential(
            nn.Linear(dim_quality, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Drift is small (2 dims), project to smaller latent or full hidden
        dim_drift_hidden = hidden_dim // 2
        self.drift_proj = nn.Sequential(
            nn.Linear(dim_drift, dim_drift_hidden),
            nn.ReLU()
        )

        # --- 3. FUSION MODULE ---
        if self.fusion_type == "quality_gated":
            self.fusion = QualityGatedAttention(hidden_dim, num_heads, dropout)
            self.fusion_out_dim = hidden_dim + dim_drift_hidden
            
        elif self.fusion_type == "standard_attn":
            self.fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.fusion_out_dim = hidden_dim + dim_drift_hidden
            
        elif self.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion_out_dim = hidden_dim + dim_drift_hidden # (We fuse vis+qual, then concat drift)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Dynamic heads from config (ModuleDict)
        self.heads = nn.ModuleDict()
        
        # config.heads should contain keys like 'ad_aigc_6s_ai' and 'ad_aigc_6s_real'
        for head_name, head_cfg in config.heads.items():
            layers = []
            curr_dim = self.fusion_out_dim
            
            # Hidden layers for this head
            for h_dim in head_cfg.hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_dim = h_dim
            
            # Final output (Output dim usually 1)
            layers.append(nn.Linear(curr_dim, head_cfg.output_dim))
            layers.append(nn.Sigmoid()) # Enforce [0,1] range for probability
            
            self.heads[head_name] = nn.Sequential(*layers)

    def forward(self, batch):
        # 1. Unpack & Cast
        v_vec = batch["visual_embedding"].float()          # [B, 512]
        q_vec = batch["quality_vector"].float()            # [B, 33]
        s_sim = batch["s_sim"].float()
        s_drift = batch["s_drift"].float()
        s_features = torch.stack([s_sim, s_drift], dim=1)  # [B, 2]

        # 2. Project
        v_emb = self.vis_proj(v_vec).unsqueeze(1)          # [B, 1, H]
        q_emb = self.qual_proj(q_vec).unsqueeze(1)         # [B, 1, H]
        s_emb = self.drift_proj(s_features).unsqueeze(1)   # [B, 1, H_drift]

        # 3. Fuse
        if self.fusion_type == "quality_gated":
            fused_main = self.fusion(content_emb=v_emb, quality_emb=q_emb)
            
        elif self.fusion_type == "standard_attn":
            attn_out, _ = self.fusion(query=v_emb, key=q_emb, value=q_emb)
            fused_main = self.norm(v_emb + attn_out)
            
        elif self.fusion_type == "concat":
            combined = torch.cat([v_emb, q_emb], dim=-1)   # [B, 1, 2H]
            fused_main = self.fusion(combined)             # [B, 1, H]

        # 4. Add Drift (Late Fusion)
        # s_emb is [B, 1, H_drift], fused_main is [B, 1, H]
        # We concatenate them to form the final representation
        final_embedding = torch.cat([fused_main, s_emb], dim=-1) # [B, 1, H + H_drift]
        final_flat = final_embedding.squeeze(1)            

        # 5. Dynamic Heads
        outputs = {}
        for head_name, head_net in self.heads.items():
            outputs[head_name] = head_net(final_flat)
            
        return outputs