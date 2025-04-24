import os
import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)
sys.path.append('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE/')

import torch
import torch.nn as nn
import e3nn.o3 as o3
from typing import Dict, Tuple, List

# Import utility functions and blocks
from holographic_vae.so3.functional import make_dict
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients
from holographic_vae.nn.blocks import CGBlock

############################################
# 1) sh_tensor_to_dict & make_vec
############################################
def sh_tensor_to_dict(sh: torch.Tensor, L: int = 7) -> Dict[int, torch.Tensor]:
    exp = (L + 1) ** 2
    if sh.size(1) != exp:
        raise ValueError(f"Expected {(L+1)**2} coeffs for L={L}, got {sh.size(1)}")
    out, idx = {}, 0
    for l in range(L + 1):
        n = 2 * l + 1
        out[l] = sh[:, idx : idx + n].unsqueeze(1)
        idx += n
    return out


def make_vec(sh_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for l in range(8):
        parts.append(sh_dict[l].squeeze(1))
    return torch.cat(parts, dim=1)

############################################
# 2) EncoderSO3 & OutputCompressor
############################################
class EncoderSO3(nn.Module):
    def __init__(self, w3j, mult: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList()
        ir_in = o3.Irreps('+'.join(f"1x{l}e" for l in range(8)))
        for _ in range(3):
            blk = CGBlock(
                irreps_in=ir_in,
                irreps_hidden=o3.Irreps('+'.join(f"{mult}x{l}e" for l in range(8))),
                w3j_matrices=w3j,
                ch_nonlin_rule='full', ls_nonlin_rule='full',
                filter_symmetric=True,
                use_batch_norm=False,
                norm_type='layer', normalization='component',
                norm_affine=True, norm_nonlinearity='swish'
            )
            self.blocks.append(blk)
            ir_in = blk.irreps_out.simplify()

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        h = x
        for blk in self.blocks:
            h = {l: t for l, t in blk(h).items() if l <= 7}
        # average over multiplicity dim=1
        return {l: t.mean(dim=1, keepdim=True) for l, t in h.items()}

class OutputCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleDict({str(l): nn.Linear(1,1,bias=False) for l in range(8)})
    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {l: self.linears[str(l)](t.transpose(1,2)).transpose(1,2) for l,t in x.items()}

############################################
# 3) Anthropometric & Embedding
############################################
class AnthropometricEncoder(nn.Module):
    def __init__(self, in_dim=25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 24), nn.ReLU(),    # increased hidden size
            nn.Linear(24, 24)
        )
    def forward(self, x):
        return self.mlp(x)

############################################
# 4) Full network with adjusted dims
############################################
class So3HyperCNNv2(nn.Module):
    def __init__(self, w3j, num_freq_bins=512, latent_dim=48, anthro_dim=25, ear_dim=12):
        super().__init__()
        self.enc  = EncoderSO3(w3j, mult=1)
        self.comp = OutputCompressor()
        self.to_z = nn.Linear(64, latent_dim)

        # conditioning inputs
        self.cond  = AnthropometricEncoder(in_dim=anthro_dim + ear_dim)
        self.f_emb = nn.Embedding(num_freq_bins, 6)    # adjusted emb dims
        self.d_emb = nn.Embedding(4, 3)
        cond_dim = 24 + 6 + 3

        # fusion MLP for concatenated [z; c]
        self.fusion_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, latent_dim),
            nn.ReLU()
        )

        # decoder: two-layer MLP
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, head, ear_feats, freq_idx, ear_idx):
        # 1) build ℓ=0 field from scalars only
        c = torch.cat([
        self.cond(torch.cat([head, ear_feats], dim=1)),  # now [B,24]
        self.f_emb(freq_idx),                            # [B, 6]
        self.d_emb(ear_idx)                              # [B, 3]
        ], dim=-1)  # → [B,33]                                         # → [B,33]
        h = {0: c.unsqueeze(-1)}                            # ℓ=0 field: [B,33,1]

        # 2) equivariant trunk
        for blk in self.enc.blocks:
            h = blk(h)

        # 3) compress back to scalars
        h = self.comp(h)                                    # still a dict {ℓ: [B,1,2ℓ+1]}

        # 4) make one big vector
        vec = make_vec(h)                                   # [B,64]

        # 5) to latent, fuse, decode
        z     = self.to_z(vec)                              # [B,latent_dim]
        # (no fusion needed here, anthro/freq/ear already used)
        out64 = self.dec(z)                                 # [B,64]

        return out64
    
if __name__ == "__main__":
    # Smoke‑test
    batch = 4
    # build w3j lookup
    w3j = get_w3j_coefficients(lmax=7)
    w3j = {k: torch.tensor(v) for k,v in w3j.items()}

    model = So3HyperCNNv2(w3j,
                          num_freq_bins=360,
                          latent_dim=48,
                          anthro_dim=25)

    # dummy inputs
    head      = torch.randn(batch, 25)
    ear_feats = torch.randn(batch, 12)
    freq_idx  = torch.randint(0, 360, (batch,))
    ear_idx   = torch.randint(0,   2, (batch,))

    out = model(head, ear_feats, freq_idx, ear_idx)
    print("Output SH shape:", out.shape)   # should be [4,64]