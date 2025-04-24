import os
import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)
sys.path.append('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE/')

import torch
import torch.nn as nn
import e3nn.o3 as o3
from typing import Dict, List
# Import utility functions and blocks
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients
from holographic_vae.nn.blocks import CGBlock

def load_w3j(lmax, device, dtype):
    tbl = get_w3j_coefficients(lmax=lmax)
    return {k: torch.tensor(v, device=device, dtype=dtype) for k, v in tbl.items()}

class EquivariantSHPredictor(nn.Module):
    """
    Anthropometry + (freq, earLR)  →  64 SH coeffs (L=7) with rotational equivariance
    """
    def __init__(self, num_freq_bins: int, L: int = 7, mul: int = 4):
        super().__init__()
        self.L, self.mul = L, mul

        # ───────── scalar encoders (exactly like before)
        self.mlp_ae = nn.Sequential(nn.Linear(25,32), nn.ReLU(), nn.Linear(32,32))
        self.emb_f  = nn.Embedding(num_freq_bins, 8)
        self.emb_lr = nn.Embedding(2, 4)
        cond_dim = 32 + 8 + 4                               # 44 scalars

        # ───────── shared Wigner 3-j tables
        dtype = torch.get_default_dtype()
        w3j   = load_w3j(L, "cpu", dtype)
        for k, t in w3j.items():
            self.register_buffer(f"w3j_{k}", t, persistent=False)
        def _w3j():  # dict view on correct device
            return {k: getattr(self, f"w3j_{k}") for k in w3j}

        # ───────── LIFT: scalars → every ℓ irreps (multiplicity = mul)
        self.lift = nn.ModuleDict({
            str(l): nn.Linear(cond_dim, mul * (2*l + 1), bias=True)
            for l in range(L + 1)
        })

        # ───────── equivariant trunk (2 CGBlocks)
        ir_hidden = o3.Irreps("+".join(f"{mul}x{l}e" for l in range(L + 1)))
        self.blocks = nn.ModuleList()
        ir = ir_hidden
        for _ in range(2):
            blk = CGBlock(
                irreps_in      = ir,
                irreps_hidden  = ir_hidden,
                w3j_matrices   = _w3j(),
                norm_type      = "component",
                normalization  = "norm",
                norm_affine    = True,
                ch_nonlin_rule = "full",
                ls_nonlin_rule = "full"
            )
            self.blocks.append(blk)

        # ───────── per-ℓ projection back to a single copy
        self.out_proj = nn.ModuleDict({
            str(l): nn.Linear(2*l + 1, 2*l + 1, bias=False) for l in range(L + 1)
        })

    # ------------------------------------------------------------------
    def forward(self, head, ear, freq_idx, ear_idx):
        B = head.size(0)
        # scalars (ℓ=0)
        scalars = torch.cat([
            self.mlp_ae(torch.cat([head, ear], dim=-1)),
            self.emb_f(freq_idx),
            self.emb_lr(ear_idx)
        ], dim=-1)                                           # [B,44]

        # LIFT to dict {ℓ: [B, mul, 2ℓ+1]}
        h = {}
        for l in range(self.L + 1):
            v = self.lift[str(l)](scalars).view(B, self.mul, 2*l + 1)
            h[l] = v

        # Equivariant mixing
        for blk in self.blocks:
            h = blk(h)

        # Average over multiplicity and project
        parts: List[torch.Tensor] = []
        for l in range(self.L + 1):
            vec = h[l].mean(1)                               # [B,2ℓ+1]
            parts.append(self.out_proj[str(l)](vec))

        return torch.cat(parts, dim=-1) 

# ---------------- smoke-test ----------------
if __name__ == "__main__":
    from config import Args
    from dataloader import create_dataloader          # ← your loader path
    torch.manual_seed(0)
    args = Args(); args.batch_size=4; args.num_workers=0
    dl,_ = create_dataloader(args)
    head, ear, fidx, eidx, sh = next(iter(dl))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = EquivariantSHPredictor(num_freq_bins=dl.dataset.F).to(device)
    out = net(head.to(device), ear.to(device), fidx.to(device), eidx.to(device))
    print("GT  :", sh.shape, "  Pred:", out.shape)
    print("✅ forward pass clean")