import torch
import torch.nn as nn

__all__ = ["BaselineSHPredictor"]

class _LinearReLU(nn.Sequential):
    """Helper: Linear → ReLU in one line."""
    def __init__(self, inp, out):
        super().__init__(nn.Linear(inp, out), nn.ReLU())

class BaselineSHPredictor(nn.Module):
    """1‑D‑Conv baseline that matches the paper’s architecture
    Input
    -----
    head : Tensor [B,13]
    ear  : Tensor [B,12]
    freq_idx : LongTensor [B]
    ear_idx  : LongTensor [B]   (0‑left, 1‑right)

    Output
    ------
    SH coefficients vector [B,64] (L = 7 truncation)
    """

    def __init__(self,
                 num_freq_bins: int,
                 ear_emb_dim: 32 = 32,
                 head_emb_dim: 32 = 32,
                 freq_emb_dim: 16 = 16,
                 lr_emb_dim:   16 = 16,
                 norm: str = "batch"):
        super().__init__()
        self.norm_name = norm.lower()

        # ───────── scalar encoders ─────────
        self.ear_enc  = _LinearReLU(12, ear_emb_dim)
        self.head_enc = _LinearReLU(13, head_emb_dim)
        self.freq_enc = nn.Embedding(num_freq_bins, freq_emb_dim)
        self.lr_enc   = nn.Embedding(2, lr_emb_dim)

        cond_dim = ear_emb_dim + head_emb_dim + freq_emb_dim + lr_emb_dim

        # affine fusion → (B, cond_dim) ► (B, C=256)
        self.fc_fuse  = _LinearReLU(cond_dim, 256)

        Norm = {"batch": nn.BatchNorm1d,
                "layer": lambda C: nn.GroupNorm(1, C),
                "instance": nn.InstanceNorm1d}[self.norm_name]

        def conv(in_ch, out_ch, k, s, last=False):
            mods = [nn.Conv1d(in_ch, out_ch, k, s)]
            if not last:
                mods += [Norm(out_ch), nn.ReLU()]
            return nn.Sequential(*mods)

        # mimic paper’s stack: (C=256, L=1) → … → (C=64, L=1)
        self.conv = nn.Sequential(
            conv(1,   4,  7, 3),
            conv(4,  16,  5, 2),
            conv(16, 32,  5, 2),
            conv(32, 32,  5, 3),
            conv(32, 64,  5, 2, last=True)   # final layer, no norm/relu
        )

    # ------------------------------------------------------------------
    def forward(self, head, ear, freq_idx, ear_idx):
        z = torch.cat([
            self.ear_enc(ear),
            self.head_enc(head),
            self.freq_enc(freq_idx),
            self.lr_enc(ear_idx)
        ], dim=-1)                         # [B, cond_dim]

        z = self.fc_fuse(z)                # [B,256]
        z = z.unsqueeze(1)                 # [B,1,256]  (N,C,L) for Conv1d
        out = self.conv(z)                 # [B,64,1]
        return out.squeeze(-1)             # → [B,64]
