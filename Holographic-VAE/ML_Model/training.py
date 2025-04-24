import os
import sys
sys.path.append('/Users/anishnair/Global_HRTF_VAE/Holographic-VAE/')

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.io as sio
import e3nn.o3 as o3
from typing import Dict, Tuple
import wandb
import random
import argparse
import warnings
import matplotlib.pyplot as plt
import math
# Set deterministic behavior for reproducibility
SEED = 46
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Import updated model and utilities
from ML_Model.dataloader import create_dataloader
from ML_Model.model import EquivariantSHPredictor
from ML_Model.baseline_model import BaselineSHPredictor
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients
from holographic_vae.so3.functional import make_vec, make_dict
from holographic_vae.utils.orthonormalization import orthonormalize_frame
from ML_Model.metrics import rotationally_averaged_lsd, directional_grad_energy

torch.autograd.set_detect_anomaly(True)


# ───────────────── reproducibility ────────────────────
SEED = 46
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ───── helper functions ───────────────────────────────────────────────────────

def inverse_SHT(sh_db: torch.Tensor, Y: torch.Tensor):
    """[B,64] dB → [B,440] linear"""
    return torch.pow(10.0, sh_db/20.0) @ Y.T

def avg_LSD(H_t: torch.Tensor, H_p: torch.Tensor, eps: float = 1e-3):
    r = (H_t+eps)/(H_p+eps)
    return (20*torch.log10(r.clamp(min=1e-10))).pow(2).mean().sqrt()

# ───── one‑fold training ──────────────────────────────────────────────────────

def train_fold(args, fold, Y, train_dl, val_dl, device, equivariant: bool):
    model_cls = EquivariantSHPredictor if equivariant else BaselineSHPredictor
    model = model_cls(num_freq_bins=train_dl.dataset.F).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=.8)
    mse   = nn.MSELoss()

    best_lsd = best_vloss = math.inf; no_imp = 0
    for epoch in range(1, args.epochs+1):
        # ─── train ───
        model.train(); tr_loss = 0.0
        for head, ear, fidx, eidx, sh in train_dl:
            head, ear, fidx, eidx, sh = [x.to(device) for x in (head, ear, fidx, eidx, sh)]
            opt.zero_grad(); pred = model(head, ear, fidx, eidx)
            loss = mse(pred, sh); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item()
        sched.step()

        # ─── evaluate ───
        model.eval(); v_mse = lsd = rot = ge = batches = 0
        with torch.no_grad():
            for bi, (head, ear, fidx, eidx, sh) in enumerate(val_dl):
                head, ear, fidx, eidx, sh = [x.to(device) for x in (head, ear, fidx, eidx, sh)]
                pred = model(head, ear, fidx, eidx)
                v_mse += mse(pred, sh).item()
                H_t, H_p = inverse_SHT(sh, Y), inverse_SHT(pred, Y)
                lsd_val  = avg_LSD(H_t, H_p).item(); lsd += lsd_val
                rot += rotationally_averaged_lsd(torch.pow(10,sh/20), torch.pow(10,pred/20),
                        L=7, K=5, measured_pts_path=os.path.join(args.measured_sh_dir,'Measured_pts.mat')).item()
                ge  += directional_grad_energy(H_p).item(); batches += 1

                # ─ plot first batch of the designated fold once after training ends ─
                if args.plot_fold == fold and epoch == 1 and bi == 0:
                    _plot_smoothness(pred[0], sh[0], fold, args)

        v_mse /= batches; lsd /= batches; rot /= batches; ge /= batches
        wandb.log({"epoch": epoch, "train_MSE": tr_loss/len(train_dl),
                   "val_MSE": v_mse, "LSD": lsd, "Rot_LSD": rot, "Grad_E": ge})
        print(f"fold {fold:02d} ▏ ep {epoch:03d} ▏ val_MSE {v_mse:.4f} ▏ LSD {lsd:.3f}")

        # checkpoint by LSD (only for the very first held‑out subject, fold 0)
        if fold == 0 and lsd < best_lsd:
            best_lsd = lsd
            ckpt = os.path.join(args.model_dir, 'best_first_subject.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"✓ saved {ckpt}  (best validation LSD for subject‑0)")
        elif lsd < best_lsd:
            best_lsd = lsd  # still track best for early‑stop but no file saved
        
        # early‑stop on val MSE
        if v_mse < best_vloss:
            best_vloss = v_mse; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience: break
    return best_lsd, rot, ge

# ───── plot helper ────────────────────────────────────────────────────────────

def _plot_smoothness(pred_vec, gt_vec, fold, args):
    plt.figure(figsize=(8,3))
    idx = torch.arange(len(pred_vec))
    plt.plot(idx, gt_vec.cpu(), label='GT', lw=1)
    plt.plot(idx, pred_vec.cpu(), label='Pred', lw=1)
    plt.title(f'SH coefficients – fold {fold} first‑batch')
    plt.xlabel('Coefficient index'); plt.ylabel('dB'); plt.legend()
    fn = os.path.join(args.model_dir, f'smoothness_fold{fold}.png')
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"✓ plot saved → {fn}")

# ───── main ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--equivariant', type=int, default=1, help='1:CG‑blocks, 0:baseline')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--model_dir', default='models')
    ap.add_argument('--plot_fold', type=int, default=0, help='Which fold to plot (‑1: none)')
    # paths
    ap.add_argument('--anthro_mat_path', default='Normalized_Anthropometric_Data.csv')
    ap.add_argument('--measured_sh_dir', default='Processed_ML/Measured')
    ap.add_argument('--Ynm_mat', default='Processed_ML/Measured/Measured_Ynm.mat')
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Y = torch.from_numpy(sio.loadmat(args.Ynm_mat)['Ynm_measured']).to(device).float()
    n_sub = pd.read_csv(args.anthro_mat_path).shape[0]

    wandb.init(project='hrtf-ablation', name='equivariant' if args.equivariant else 'baseline', config=vars(args))
    all_scores = []
    for fold in range(n_sub):
        print(f"\n─── fold {fold}/{n_sub-1} held‑out ───")
        args.val_idx = fold
        train_dl, val_dl = create_dataloader(args, seed=SEED)
        best = train_fold(args, fold, Y, train_dl, val_dl, device, bool(args.equivariant))
        all_scores.append(best)

    scores = np.array(all_scores)
    print("\n==== Leave‑one‑out summary ====")
    print(f"mean LSD      : {scores[:,0].mean():.3f} dB")
    print(f"mean Rot‑LSD  : {scores[:,1].mean():.3f} dB")
    print(f"mean Grad‑E   : {scores[:,2].mean():.3f}")
    wandb.finish()

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
