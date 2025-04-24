"""
Evaluation & Visualization

Fig 1 – Cumulative RMS-LSD vs Frequency  
Fig 2 – Error Spectrum (Mean LSD per Frequency)  

Dataset : HUTUBS (leave-one-out validation)  
Model   : model2_lsd.pth – SO(3)-equivariant CNN + FiLM  
Metric  : log-spectral distortion (LSD)
"""

import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import sys

# ── 1) Setup paths ─────────────────────────────────────────────────────────────
ROOT = Path('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE')
sys.path.append(str(ROOT))

# ── 2) Imports & model config ─────────────────────────────────────────────────
MODEL_PTH = ROOT / 'ML_Model_2' / 'model2_lsd.pth'

from ML_Model_2.dataloader import create_dataloader, Args
from ML_Model_2.model     import So3HyperCNNv2 as So3HyperCNN
from ML_Model_2.training  import (
    compute_sh_basis,
    inverse_sh_transform,
    compute_global_stats_from_loader,
)
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients

# ── 3) Constants ───────────────────────────────────────────────────────────────
FS           = 44100
NUM_BINS     = 512
SH_ORDER     = 7
SPATIAL_BINS = 256
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS          = 1e-6

# ── 4) Load data ───────────────────────────────────────────────────────────────
args = Args()
args.device             = DEVICE
args.measured_hrtf_dir  = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
args.measured_sh_dir    = args.measured_hrtf_dir

train_loader, val_loader = create_dataloader(args)

# ── 5) Compute global de‑normalization stats ──────────────────────────────────
g_mean, g_std = compute_global_stats_from_loader(train_loader, DEVICE)
# add batch dim so we can broadcast later
g_mean = g_mean.unsqueeze(0)  # [1, C]
g_std  = g_std .unsqueeze(0)  # [1, C]

# ── 6) Load & prepare model ──────────────────────────────────────────────────
# Wigner 3j coeffs for the equivariant blocks
w3j = {
    k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
    for k, v in get_w3j_coefficients(lmax=SH_ORDER).items()
}

model = So3HyperCNN(
    w3j,
    latent_dim       = 128,
    num_freq_bins    = NUM_BINS,
    anthro_dim       = 25,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()

# ── 7) Precompute SH basis ───────────────────────────────────────────────────
Y = compute_sh_basis(SH_ORDER, SPATIAL_BINS, DEVICE)  # [spatial_bins, (L+1)^2]

# ── 8) Evaluate over validation set ──────────────────────────────────────────
records = []  # will hold (freq_bin, lsd_value)

with torch.no_grad():
    for ear_a, head_a, _, sh, _, freq_idx, dom_idx in val_loader:
        # --- 8.1) flatten / squeeze ---
        # sh: [B, S, maybe 1] → [B, S]
        if sh.dim() == 3:
            sh = sh.mean(-1)
        else:
            sh = sh.squeeze(1).mean(-1)
        sh = sh.to(DEVICE)

        # ear_a / head_a: [B, D, 1?] → [B, D]
        ear_a  = ear_a .view(ear_a .size(0), -1).to(DEVICE)
        head_a = head_a.view(head_a.size(0), -1).to(DEVICE)

        # freq_idx / dom_idx: may be one‑hot or already index
        if freq_idx.dim() > 1:
            freq_idx = freq_idx.argmax(dim=1)
        if dom_idx.dim() > 1:
            dom_idx  = dom_idx .argmax(dim=1)
        freq_idx = freq_idx.to(DEVICE).long()
        dom_idx  = dom_idx .to(DEVICE).long()

        # --- 8.2) normalize, forward, de‑normalize ---
        sh_n  = (sh - g_mean) / (g_std + EPS)
        recon = model(sh_n, head_a, ear_a, freq_idx, dom_idx)
        recon = recon * g_std + g_mean

        # --- 8.3) inverse‐SHT to get spatial magnitudes ---
        H_t = inverse_sh_transform(sh,    Y)  # [B, spatial_bins]
        H_r = inverse_sh_transform(recon, Y)  # [B, spatial_bins]

        # --- 8.4) per‐bin LSD, then mean over space to get one LSD per sample ---
        lsd_bins = 20.0 * torch.log10(
            torch.clamp((H_t + EPS) / (H_r + EPS), min=EPS)
        ).abs()                                 # [B, spatial_bins]
        lsd_mean = lsd_bins.mean(dim=1)         # [B]

        # --- 8.5) record (freq_idx, lsd) pairs ---
        for f, l in zip(freq_idx.cpu().numpy(), lsd_mean.cpu().numpy()):
            records.append((int(f), float(l)))

# ── 9) Aggregate per freq‐bin & compute cumulative RMS‐LSD ─────────────────
df       = pd.DataFrame(records, columns=['freq_bin','lsd'])
mean_lsd = df.groupby('freq_bin')['lsd'].mean().sort_index()

# cumulative RMS‐LSD up to each freq
sq_cum      = (mean_lsd**2).cumsum()
n_freqs     = np.arange(1, len(mean_lsd)+1)
cum_rms_lsd = np.sqrt(sq_cum / n_freqs)

# map bin → frequency in kHz
hz_per_bin = (FS/2) / (NUM_BINS - 1)
freq_khz   = mean_lsd.index * hz_per_bin / 1000.0

# ── 10) Plot 1: Cumulative RMS‑LSD ──────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(    freq_khz, cum_rms_lsd, lw=2)
plt.axvline(6, color='gray', ls='--', lw=0.8)
plt.text(   6.1, cum_rms_lsd.iloc[-1]*0.9, "6 kHz", color='gray')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Cumulative RMS‑LSD (dB)")
plt.title("SO(3)+FiLM – Cumulative RMS‑LSD vs Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 11) Plot 2: Error Spectrum ──────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(    freq_khz, mean_lsd, lw=2)
plt.axvline(6, color='gray', ls='--', lw=0.8)
plt.text(   6.1, mean_lsd.max()*0.9, "6 kHz", color='gray')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Mean LSD per Frequency Bin (dB)")
plt.title("SO(3)+FiLM – LSD Error Spectrum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()