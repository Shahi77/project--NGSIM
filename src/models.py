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

class MultichannelAttention(nn.Module):
    """
    Paper's multichannel attention: separate channels for different feature types,
    then adaptive weighting
    """
    def __init__(self, in_dims_list, d_model):
        super().__init__()
        self.projections = nn.ModuleList()
        self.attention_weights = nn.ModuleList()
        
        for in_dim in in_dims_list:
            # Separate projection and attention weight computation
            self.projections.append(nn.Linear(in_dim, d_model))
            self.attention_weights.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ))
        
        self.fusion = nn.Linear(d_model * len(in_dims_list), d_model)
    
    def forward(self, *inputs):
        """inputs: list of tensors (B, T, C_i) - all should have same B and T"""
        weighted = []
        for inp, proj, att_weight in zip(inputs, self.projections, self.attention_weights):
            # Project to d_model
            projected = proj(inp)  # (B, T, d_model)
            # Compute attention weights
            weights = att_weight(projected)  # (B, T, d_model)
            # Apply attention
            weighted_feat = projected * weights  # (B, T, d_model)
            weighted.append(weighted_feat)
        
        # Concatenate and fuse
        concat = torch.cat(weighted, dim=-1)  # (B, T, d_model * num_channels)
        return self.fusion(concat)  # (B, T, d_model)

class ImprovedTrajectoryTransformer(nn.Module):
    """
    Architecture closer to paper:
    1. Multichannel attention for vehicle dynamics, spatial relations, map features
    2. Transformer encoder for sequence modeling
    3. Decoder that outputs ABSOLUTE positions (not deltas) using autoregressive attention
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Feature projections
        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)
        
        # Multichannel attention (paper Eq. 1-4)
        self.multi_att = MultichannelAttention([d_model, d_model, d_model, d_model], d_model)
        
        # Neighbor aggregation with attention
        self.neigh_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder (paper uses 8 heads as mentioned)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder: autoregressive with cross-attention to encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head: predict ABSOLUTE position at each step
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        
        # Learnable query embeddings for decoder
        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, last_obs_pos=None, pred_len=None):
        """
        target: (B, T_obs, 7)
        neigh_dyn: (B, K, T_obs, 7)
        neigh_spatial: (B, K, T_obs, 18)
        lane: (B, T_obs, 1) or (B, 1)
        last_obs_pos: (B, 2)
        pred_len: override prediction length
        """
        B, T_obs = target.size(0), target.size(1)
        current_pred_len = pred_len if pred_len is not None else self.pred_len

        # --- Fix neighbor tensor shapes ---
        # Expected: (B, K, T_obs, features)
        if neigh_dyn.dim() == 4:
            if neigh_dyn.size(2) == T_obs:
                # Already (B, K, T_obs, 7)
                pass
            elif neigh_dyn.size(1) == T_obs:
                # (B, T_obs, K, 7) -> (B, K, T_obs, 7)
                neigh_dyn = neigh_dyn.permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unexpected neigh_dyn shape: {neigh_dyn.shape}")
        else:
            raise ValueError(f"Invalid neigh_dyn dimensions (expected 4D): {neigh_dyn.shape}")

        if neigh_spatial.dim() == 4:
            if neigh_spatial.size(2) == T_obs:
                pass
            elif neigh_spatial.size(1) == T_obs:
                neigh_spatial = neigh_spatial.permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unexpected neigh_spatial shape: {neigh_spatial.shape}")
        else:
            raise ValueError(f"Invalid neigh_spatial dimensions (expected 4D): {neigh_spatial.shape}")

        K = neigh_dyn.size(1)

        # --- Fix lane tensor shape ---
        # Expected: (B, T_obs, 1)
        if lane.dim() == 2:
            if lane.size(1) == 1:
                # (B, 1) -> (B, T_obs, 1)
                lane = lane.unsqueeze(1).expand(-1, T_obs, -1)
            elif lane.size(1) == T_obs:
                # (B, T_obs) -> (B, T_obs, 1)
                lane = lane.unsqueeze(-1)
            else:
                raise ValueError(f"Unexpected lane shape: {lane.shape}")
        elif lane.dim() == 3:
            # Check if we need to expand time dimension
            if lane.size(1) == 1 and lane.size(1) != T_obs:
                # (B, 1, 1) -> (B, T_obs, 1)
                lane = lane.expand(-1, T_obs, -1)
            elif lane.size(1) != T_obs:
                raise ValueError(f"Lane time dimension {lane.size(1)} doesn't match T_obs {T_obs}")
        else:
            raise ValueError(f"Invalid lane dimensions: {lane.shape}")

        # --- Project features ---
        target_feat = self.target_proj(target)  # (B, T_obs, d_model)
        lane_feat = self.lane_proj(lane)        # (B, T_obs, d_model)

        # --- Neighbor aggregation ---
        # Reshape: (B, K, T_obs, 7) -> (B*K, T_obs, 7)
        neigh_dyn_reshaped = neigh_dyn.view(B * K, T_obs, 7)
        neigh_dyn_proj = self.neigh_dyn_proj(neigh_dyn_reshaped)  # (B*K, T_obs, d_model)
        neigh_dyn_proj = neigh_dyn_proj.view(B, K, T_obs, self.d_model)
        neigh_dyn_agg = neigh_dyn_proj.mean(dim=1)  # (B, T_obs, d_model)

        neigh_spatial_reshaped = neigh_spatial.view(B * K, T_obs, 18)
        neigh_spatial_proj = self.neigh_spatial_proj(neigh_spatial_reshaped)
        neigh_spatial_proj = neigh_spatial_proj.view(B, K, T_obs, self.d_model)
        neigh_spatial_agg = neigh_spatial_proj.mean(dim=1)  # (B, T_obs, d_model)

        # --- Multichannel attention fusion ---
        fused = self.multi_att(target_feat, neigh_dyn_agg, neigh_spatial_agg, lane_feat)
        fused = self.pos_enc(fused)
        memory = self.encoder(fused)

        # --- Decoder ---
        queries = self.query_embed.weight[:current_pred_len].unsqueeze(0).repeat(B, 1, 1)
        queries = self.pos_embed_dec(queries)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_pred_len).to(queries.device)

        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)
        preds = self.output_head(decoded)  # (B, T_pred, 2)

        if last_obs_pos is not None:
            preds = preds + last_obs_pos.unsqueeze(1)

        return preds 