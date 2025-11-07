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
    def __init__(self, in_dims_list, d_model):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims_list])
        self.attention_weights = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Sigmoid())
            for _ in in_dims_list
        ])
        self.fusion = nn.Linear(d_model * len(in_dims_list), d_model)

    def forward(self, *inputs):
        weighted = []
        for inp, proj, att in zip(inputs, self.projections, self.attention_weights):
            p = proj(inp)
            w = att(p)
            weighted.append(p * w)
        return self.fusion(torch.cat(weighted, dim=-1))


class ImprovedTrajectoryTransformer(nn.Module):
    """
    Residual decoder over constant-velocity baseline in the road-aligned frame.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8, dt=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.dt = dt

        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)

        self.multi_att = MultichannelAttention([d_model, d_model, d_model, d_model], d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)

        # residual head (predict deltas on top of CV baseline)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def _cv_baseline(self, last_pos, last_vel, T):
        # last_pos (B,2), last_vel (B,2)
        steps = torch.arange(1, T + 1, device=last_pos.device, dtype=last_pos.dtype).unsqueeze(0).unsqueeze(-1)  # (1,T,1)
        incr = last_vel.unsqueeze(1) * steps * self.dt  # (B,T,2)
        return last_pos.unsqueeze(1) + incr  # (B,T,2)

    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        """
        target: (B, T_obs, 7) normalized
        neigh_dyn: (B, K, T_obs, 7)
        neigh_spatial: (B, K, T_obs, 18)
        lane: (B, 1) or (B, T_obs, 1)
        """
        B, T_obs = target.size(0), target.size(1)
        T_pred = pred_len if pred_len is not None else self.pred_len
        K = neigh_dyn.size(1)

        # lane shape to (B, T_obs, 1)
        if lane.dim() == 2:
            lane = lane.unsqueeze(1).expand(-1, T_obs, -1)
        elif lane.dim() == 3 and lane.size(1) == 1:
            lane = lane.expand(-1, T_obs, -1)

        # project
        tgt_feat = self.target_proj(target)           # (B,T_obs,d)
        lane_feat = self.lane_proj(lane)              # (B,T_obs,d)

        nd = neigh_dyn.view(B * K, T_obs, 7)
        nd = self.neigh_dyn_proj(nd).view(B, K, T_obs, self.d_model).mean(dim=1)

        ns = neigh_spatial.view(B * K, T_obs, 18)
        ns = self.neigh_spatial_proj(ns).view(B, K, T_obs, self.d_model).mean(dim=1)

        fused = self.multi_att(tgt_feat, nd, ns, lane_feat)
        memory = self.encoder(self.pos_enc(fused))

        queries = self.query_embed.weight[:T_pred].unsqueeze(0).repeat(B, 1, 1)
        queries = self.pos_embed_dec(queries)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_pred).to(queries.device)

        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)  # (B,T_pred,d)
        residual = self.residual_head(decoded)                      # (B,T_pred,2)

        last_pos = target[:, -1, :2]
        # prefer velocity channel if available and finite, else finite difference
        last_vel = target[:, -1, 2:4]
        cv = self._cv_baseline(last_pos, last_vel, T_pred)          # (B,T_pred,2)

        return cv + residual, cv
