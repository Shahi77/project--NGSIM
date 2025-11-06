"""
models.py - ImprovedTrajectoryTransformer
- increased capacity: d_model=384, nhead=12
- dropout
- robust multichannel fusion and neighbor aggregation
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultichannelFusion(nn.Module):
    """
    Projects each channel to d_model, applies learnable gating and fuses.
    """
    def __init__(self, in_dims, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(d_model, d_model),
                                                  nn.Sigmoid()) for _ in in_dims])
        self.fusion = nn.Linear(d_model * len(in_dims), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *inputs):
        # inputs: (B, T, C_i) - if a channel has C_i==1 and T==1, it should be expanded before calling
        outs = []
        for x, p, g in zip(inputs, self.proj, self.gates):
            # ensure shape (B, T, C)
            proj = p(x)    # (B, T, d_model)
            gate = g(proj) # (B, T, d_model)
            outs.append(proj * gate)
        concat = torch.cat(outs, dim=-1)
        return self.dropout(self.fusion(concat))

class ImprovedTrajectoryTransformer(nn.Module):
    def __init__(self, d_model=384, nhead=12, num_layers=4, pred_len=25, k_neighbors=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k_neighbors = k_neighbors

        # Projectors for input feature dims
        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)

        # Fusion
        self.fusion = MultichannelFusion([d_model, d_model, d_model, d_model], d_model, dropout=dropout)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=4*d_model, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=4*d_model, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        # Learnable queries
        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)

    def forward(self, target, neigh_dyn, neigh_spatial, lane, last_obs_pos=None, pred_len=None):
        """
        target: (B, T_obs, 7)
        neigh_dyn: (B, K, T_obs, 7)  OR (B, K, T_obs, 7)
        neigh_spatial: same pattern (B, K, T_obs, 18)
        lane: (B, 1) or (B, T_obs, 1)
        """
        B, T_obs = target.shape[0], target.shape[1]
        current_pred_len = pred_len if pred_len is not None else self.pred_len

        # Normalize lane shape to (B, T_obs, 1)
        if lane.dim() == 2 and lane.size(1) == 1:
            lane = lane.unsqueeze(1).expand(-1, T_obs, -1)  # (B, T_obs, 1)
        elif lane.dim() == 2 and lane.size(1) == T_obs:
            lane = lane.unsqueeze(-1)
        elif lane.dim() == 3 and lane.size(1) == 1:
            lane = lane.expand(-1, T_obs, -1)

        # neighbor handling
        if neigh_dyn.dim() == 4:
            # (B, K, T_obs, features)
            pass
        else:
            raise ValueError("neigh_dyn must be 4D (B,K,T,feat)")

        B, K, T_check, f = neigh_dyn.shape
        if T_check != T_obs:
            raise ValueError(f"neigh time dim {T_check} != target time dim {T_obs}")

        # project target & lane
        target_feat = self.target_proj(target)  # (B, T_obs, d_model)
        lane_feat = self.lane_proj(lane)        # (B, T_obs, d_model)

        # neighbor aggregation: project per neighbor and average
        neigh_dyn_flat = neigh_dyn.view(B * K, T_obs, neigh_dyn.shape[-1])
        neigh_dyn_proj = self.neigh_dyn_proj(neigh_dyn_flat)  # (B*K, T, d_model)
        neigh_dyn_proj = neigh_dyn_proj.view(B, K, T_obs, self.d_model).mean(dim=1)  # (B, T, d_model)

        neigh_spatial_flat = neigh_spatial.view(B * K, T_obs, neigh_spatial.shape[-1])
        neigh_spatial_proj = self.neigh_spatial_proj(neigh_spatial_flat)
        neigh_spatial_proj = neigh_spatial_proj.view(B, K, T_obs, self.d_model).mean(dim=1)

        # fusion
        fused = self.fusion(target_feat, neigh_dyn_proj, neigh_spatial_proj, lane_feat)  # (B, T, d_model)
        fused = self.pos_enc(fused)
        memory = self.encoder(fused)  # (B, T, d_model)

        # decoder queries
        queries = self.query_embed.weight[:current_pred_len].unsqueeze(0).repeat(B, 1, 1)  # (B, T_pred, d_model)
        queries = self.pos_embed_dec(queries)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_pred_len).to(queries.device)

        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)
        preds = self.output_head(decoded)  # (B, T_pred, 2)

        if last_obs_pos is not None:
            preds = preds + last_obs_pos.unsqueeze(1)

        return preds
