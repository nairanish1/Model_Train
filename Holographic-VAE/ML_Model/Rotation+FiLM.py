"""
Fig 1 – Cumulative RMS-LSD vs Frequency
Fig 2 – Error Spectrum (Mean LSD per Frequency)

Dataset : HUTUBS   – leave‑one‑out validation subjects
Model   : 128_best_hrtf_cnn_ms_lsd.pth   – SO(3)‑equivariant CNN + FiLM
Metric  : log‑spectral distortion (LSD)
"""

import sys, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------- paths / imports
ROOT      = Path('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE')
MODEL_PTH = ROOT / 'ML_Model' / '128_best_hrtf_cnn_ms_lsd.pth'
sys.path.append(str(ROOT))

from ML_Model.dataloader import create_dataloader
from config                import Args
from ML_Model.model        import So3HyperCNN
from ML_Model.training     import (
    compute_sh_basis,
    inverse_sh_transform,
    compute_global_stats_from_loader,
)
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients

# ------------------------------------------------------------ constants
FS        = 44100              # HUTUBS sample rate
NUM_BINS  = 512                # num_freq_bins
SH_ORDER  = 7
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS       = 1e-6

# ------------------------------------------------------------ dataloaders
args = Args()
args.device            = DEVICE
args.measured_hrtf_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
args.measured_sh_dir   = args.measured_hrtf_dir

train_loader, val_loader = create_dataloader(args)

# global mean/std of SH coeffs for de‑normalization
g_mean, g_std = compute_global_stats_from_loader(train_loader, DEVICE)
g_mean, g_std = g_mean.unsqueeze(0), g_std.unsqueeze(0)   # shape [1, C]

# ------------------------------------------------------------ model
w3j = {k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
       for k, v in get_w3j_coefficients(lmax=SH_ORDER).items()}

model = So3HyperCNN(
    w3j,
    latent_dim       = 128,
    num_freq_bins    = NUM_BINS,
    anthro_input_dim = 25,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()

Y = compute_sh_basis(SH_ORDER, 256, DEVICE)  # SH basis cache

# ------------------------------------------------------------ evaluation
records = []   # will hold (freq_bin_idx, mean_spatial_lsd)

with torch.no_grad():
    for ear_a, head_a, _, sh, _, freq_idx, dom_idx in val_loader:
        # flatten SH to [B, C]
        sh = sh.mean(-1) if sh.dim()==3 else sh.squeeze(1).mean(-1)
        sh = sh.to(DEVICE)

        # normalize & forward
        sh_n = (sh - g_mean) / (g_std + EPS)
        recon = model(
            sh_n,
            head_a.to(DEVICE),
            ear_a.to(DEVICE),
            freq_idx.to(torch.long).to(DEVICE),
            dom_idx .to(torch.long).to(DEVICE),
        )
        recon = recon * g_std + g_mean    # de‑normalize

        # reconstruct HRTF magnitudes over spatial bins
        H_t = inverse_sh_transform(sh,    Y)  # [B, spatial_bins]
        H_r = inverse_sh_transform(recon, Y)  # [B, spatial_bins]

        # compute LSD per spatial bin
        lsd_bins = 20 * torch.log10(
            torch.clamp((H_t + EPS)/(H_r + EPS), min=EPS)
        ).abs()                                # [B, spatial_bins]

        # now average over spatial bins → one scalar per sample
        lsd_mean_spatial = lsd_bins.mean(dim=1)  # [B]

        # record (freq_bin, lsd)
        for f, e in zip(freq_idx.cpu().numpy().squeeze(),
                        lsd_mean_spatial.cpu().numpy()):
            records.append((int(f), float(e)))

# ------------------------------------------------------------ aggregate
df        = pd.DataFrame(records, columns=['freq_bin','lsd'])
mean_lsd  = df.groupby('freq_bin')['lsd'] \
              .mean() \
              .sort_index()
# cumulative RMS‑LSD: √( Σ (mean_lsd²) / N_freqs_up_to_here )
sq_cum      = (mean_lsd**2).cumsum()
n_freqs     = np.arange(1, len(mean_lsd)+1)
cum_rms_lsd = np.sqrt(sq_cum / n_freqs)

# convert freq_bin → frequency (kHz)
hz_per_bin = (FS/2) / (NUM_BINS-1)
freq_khz   = mean_lsd.index * hz_per_bin / 1_000.0

# ------------------------------------------------------------ plots
plt.figure(figsize=(8,4))
plt.plot(freq_khz, cum_rms_lsd, lw=2)
plt.axvline(6, color='gray', ls='--', lw=0.8)
plt.text(6.1, cum_rms_lsd.iloc[-1]*0.9, "6 kHz", color='gray')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Cumulative RMS‑LSD (dB)")
plt.title("SO(3)+FiLM – Cumulative RMS‑LSD vs Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(freq_khz, mean_lsd, lw=2)
plt.axvline(6, color='gray', ls='--', lw=0.8)
plt.text(6.1, mean_lsd.max()*0.9, "6 kHz", color='gray')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Mean LSD per Frequency Bin (dB)")
plt.title("SO(3)+FiLM – Error Spectrum")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
