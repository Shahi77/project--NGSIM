import torch
import torch.nn as nn
import math

# ---------------- LSTM BASELINE ----------------
class SimpleSocialLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=2, pred_len=25, k_neighbors=8):
        super().__init__()
        self.pred_len = pred_len
        self.k = k_neighbors
        self.hidden_dim = hidden_dim

        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.neigh_enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.init_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dec = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

        # initialize forget gate bias to 1.0
        for name, param in self.enc.named_parameters():
            if "bias" in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        B = target.shape[0]
        pred_len = self.pred_len if pred_len is None else pred_len

        _, (h_t, _) = self.enc(target)
        h_t = h_t.squeeze(0)
        neigh_flat = neigh_dyn.view(-1, neigh_dyn.shape[2], neigh_dyn.shape[3])
        _, (h_n, _) = self.neigh_enc(neigh_flat)
        h_n = h_n.squeeze(0).view(B, self.k, -1).mean(1)
        h0 = torch.tanh(self.init_fc(torch.cat([h_t, h_n], -1))).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        preds, inp, state = [], torch.zeros(B, 1, 2, device=target.device), (h0, c0)
        for _ in range(pred_len):
            out, state = self.dec(inp, state)
            step = self.out(out)
            preds.append(step)
            inp = step
        return torch.cat(preds, 1)


# ---------------- TRANSFORMER ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class MultichannelAttention(nn.Module):
    def __init__(self, in_dims_list, d_model):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in in_dims_list])
        self.att = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Tanh())
            for _ in in_dims_list
        ])
        self.fuse = nn.Linear(d_model * len(in_dims_list), d_model)
    def forward(self, *inputs):
        outs = []
        for x, p, a in zip(inputs, self.proj, self.att):
            z = p(x)
            outs.append(z * a(z))
        return self.fuse(torch.cat(outs, -1))


class ImprovedTrajectoryTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8, dt=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.dt = dt

        # --- Encoders ---
        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)
        self.multi_att = MultichannelAttention([d_model]*4, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # --- Transformer blocks ---
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, 0.1, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, 4*d_model, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        # --- Query & output heads ---
        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 2)
        )

    def _cv_baseline(self, last_pos, last_vel, T):
        steps = torch.arange(1, T+1, device=last_pos.device).view(1, T, 1)
        return last_pos.unsqueeze(1) + last_vel.unsqueeze(1) * steps * self.dt

    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None, cv_warmup_alpha=0.0):
        """
        Forward pass with optional CV residual warmup.
        cv_warmup_alpha ∈ [0,1]: blends output toward CV baseline in early epochs.
        """
        B, T = target.size(0), target.size(1)
        T_pred = pred_len or self.pred_len
        # --- normalize lane shape to (B, T, 1)
        if lane.dim() == 2:                 # (B, 1)
            lane = lane.unsqueeze(1).expand(-1, T, -1)
        elif lane.dim() == 3 and lane.size(1) == 1:  # (B, 1, 1)
            lane = lane.expand(-1, T, -1)

        # --- feature projections ---
        tgt = self.target_proj(target)
        lane_f = self.lane_proj(lane)
        K = neigh_dyn.size(1)
        nd = self.neigh_dyn_proj(neigh_dyn.view(B*K, T, 7)).view(B, K, T, -1).mean(1)
        ns = self.neigh_spatial_proj(neigh_spatial.view(B*K, T, 18)).view(B, K, T, -1).mean(1)

        fused = self.multi_att(tgt, nd, ns, lane_f)
        memory = self.encoder(self.pos_enc(fused))
        memory = nn.LayerNorm(memory.size(-1), elementwise_affine=False).to(memory.device)(memory)

        # --- decode ---
        queries = self.pos_embed_dec(self.query_embed.weight[:T_pred].unsqueeze(0).repeat(B,1,1))
        decoded = self.decoder(queries, memory)
        residual = self.residual_head(decoded)

        # --- constant velocity baseline ---
        last_pos, last_vel = target[:, -1, :2], target[:, -1, 2:4]
        cv = self._cv_baseline(last_pos, last_vel, T_pred)

        # --- optional CV warmup blend ---
        if cv_warmup_alpha > 0:
            pred = cv + residual
            pred = (1 - cv_warmup_alpha) * pred + cv_warmup_alpha * cv
            return pred, cv
        else:
            return cv + residual, cv
