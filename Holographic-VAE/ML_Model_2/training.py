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
# Set deterministic behavior for reproducibility
SEED = 46
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Import updated model and utilities
from ML_Model_2.dataloader import create_dataloader, Args
from ML_Model_2.model import So3HyperCNNv2 as So3HyperCNN
from ML_Model_2.model import sh_tensor_to_dict
from holographic_vae.cg_coefficients.get_w3j_coefficients import get_w3j_coefficients
from holographic_vae.so3.functional import make_vec, make_dict
from holographic_vae.utils.orthonormalization import orthonormalize_frame
from ML_Model_2.metrics import rotationally_averaged_lsd, directional_grad_energy

torch.autograd.set_detect_anomaly(True)


def load_w3j_matrices(lmax: int, device: torch.device) -> Dict[Tuple[int, int, int], torch.Tensor]:
    w3j_numpy = get_w3j_coefficients(lmax=lmax)
    return {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in w3j_numpy.items()}


def deterministic_loss(recon: torch.Tensor, target: torch.Tensor, recon_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the weighted MSE loss, after replacing any NaNs or infinities in the inputs.
    """
    # Replace any NaN or infinite values in both recon and target
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    mse_loss = F.mse_loss(recon, target, reduction='mean')
    return recon_weight * mse_loss, mse_loss

def compute_sh_basis(L: int, S: int, device: torch.device) -> torch.Tensor:
    N = int(np.sqrt(S))
    az = torch.linspace(0, 2 * np.pi * (1 - 1/N), N, device=device)
    el = torch.linspace(0, np.pi, N, device=device)
    azg, elg = torch.meshgrid(az, el, indexing='ij')
    pts = torch.stack([
        torch.sin(elg).flatten() * torch.cos(azg).flatten(),
        torch.sin(elg).flatten() * torch.sin(azg).flatten(),
        torch.cos(elg).flatten()
    ], dim=1)
    Y = torch.cat([
        o3.spherical_harmonics(l, pts, normalize='component').real
        for l in range(L+1)
    ], dim=1)
    return Y


def inverse_sh_transform(sh_db: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # sh_db is in dB; first convert to linear:
    sh_lin = torch.pow(10.0, sh_db / 20.0)
    # then reconstruct the magnitude at each direction:
    return sh_lin @ Y.T


def compute_avg_lsd(H_target: torch.Tensor, H_recon: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Global log‑spectral distortion:
      20 * mean( | log10((H_target + eps)/(H_recon + eps)) | )
    with NaNs and infs clamped to eps for stability.
    """
    # clamp out any NaNs or ±inf
    H_target = torch.nan_to_num(H_target, nan=eps, posinf=eps, neginf=eps)
    H_recon  = torch.nan_to_num(H_recon,  nan=eps, posinf=eps, neginf=eps)
    # compute ratio in linear domain
    ratio = (H_target + eps) / (H_recon + eps)
    # avoid log10(0)
    ratio = ratio.clamp(min=1e-10)
    # log‑spectral distortion
    lsd = 20.0 * torch.log10(ratio)
    return torch.sqrt((lsd.pow(2)).mean())



#def compute_global_stats_from_loader(loader: DataLoader, device: torch.device):
    #"""
    #Computes global mean and std of SH coefficients over the entire training set.
    #Assumes that each batch returned contains a tensor 'sh' of shape [B, num_coeffs].
    #"""
    #total_sum = None
    #total_squared_sum = None
    #total_count = 0
    #for ear_a, head_a, _, sh, subj, freq, dom in loader:
        # Make sure sh is of shape [B, num_coeff] by reducing extra dims.
        #if sh.dim() == 3:
            #sh = sh.mean(dim=-1)
        #elif sh.dim() == 4:
           #sh = sh.squeeze(1).mean(dim=-1)
        #sh = sh.float().to(device)
       # if total_sum is None:
            #total_sum = sh.sum(dim=0)
            #total_squared_sum = (sh ** 2).sum(dim=0)
        #else:
            #total_sum += sh.sum(dim=0)
            #total_squared_sum += (sh ** 2).sum(dim=0)
        #total_count += sh.size(0)
    #global_mean = total_sum / total_count
    #global_std = torch.sqrt(total_squared_sum / total_count - global_mean ** 2 + 1e-5)
    #return global_mean, global_std

# Ensure DataLoader workers are seeded for reproducibility
def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def train_model(args, num_epochs, lr, patience,
                lsd_weight, lsd_warmup_epochs,recon_weight): 
    device = args.device
    w3j = load_w3j_matrices(7, device)
    best_val_loss = float('inf')
    best_ms_lsd = float('inf')
    best_rot_lsd  = float('inf')
    best_grad_e   = float('inf')
    epochs_no_improve = 0
    
    # this .mat should contain variable SH_Vec_matrix of shape [440,64]
    
    data = sio.loadmat(args.shvec_path)
    Y = torch.from_numpy(data['Ynm_measured']).to(device).float()

    measured_pts_path = os.path.join(
        args.measured_sh_dir,
        'Measured_pts.mat'
    )

    wandb.init(project="hrtf-ablation",
               entity="nairanish-georgia-institute-of-technology",
               config={
                   "num_epochs": num_epochs,
                   "learning_rate": lr,
                   "patience": patience,
                   "lsd_weight": lsd_weight,
                   "lsd_warmup_epochs": lsd_warmup_epochs,
                   "recon_weight": recon_weight,
                   "batch_size": args.batch_size,
                   "num_freq_bins": args.num_freq_bins,
                   "latent_dim": 128 # Updated latent dimension.
               })
    config = wandb.config
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader, val_loader = create_dataloader(args, seed=SEED)
    device = args.device
    w3j = load_w3j_matrices(7, device)

    # Instantiate model with latent_dim = 128.
    model = So3HyperCNN(w3j,
                      latent_dim   = 128,
                      num_freq_bins= args.num_freq_bins,
                      anthro_dim   = 25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.8
)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", total_params)
    wandb.log({"Total trainable parameters": total_params})


    for epoch in range(num_epochs):
        # Dynamic LSD Weight Warmup: ramp up until lsd_warmup_epochs.
        if (epoch + 1) < lsd_warmup_epochs:
            current_lsd_weight = lsd_weight * ((epoch + 1) / lsd_warmup_epochs)
        else:
            current_lsd_weight = lsd_weight

        model.train()
        running_loss = 0.0
        epoch_mse_total = 0.0
        batch_count = 0

        for ear_a, head_a, _, sh, subj, freq, dom in train_loader:
            if dom.dim() > 1:
                    dom_idx = dom.argmax(dim=1).long()
            else:                           # already an index
                dom_idx = dom.long()
            # Ensure sh is reduced to shape [B, num_coeff].
            ear_a, head_a = ear_a.to(device), head_a.to(device)
            freq, dom  = freq.to(device), dom.to(device)
            subj = subj.long().to(device).squeeze()
            if sh.dim() == 3:
                sh = sh.mean(dim=-1)
            elif sh.dim() == 4:
                sh = sh.squeeze(1).mean(dim=-1)
            sh = sh.to(device)
            # Apply global normalization.
            # global_mean and global_std are of shape [num_coeff], so unsqueeze for batch broadcasting.
            # Move remaining tensors to device

            optimizer.zero_grad()
            # The deterministic model returns only the reconstruction.
            recon = model(sh, head_a, ear_a, freq, dom_idx)

            loss_recon, mse_val = deterministic_loss(recon, sh, recon_weight)
            epoch_mse_total += mse_val.item()
            batch_count += 1

            # Denormalize reconstruction using global stats.
            sh_lin    = torch.pow(10.0, sh    / 20.0)  # [B,64]
            recon_lin = torch.pow(10.0, recon / 20.0)  # [B,64]
            H_target  = sh_lin    @ Y.T               # [B,440]
            H_recon   = recon_lin @ Y.T               # [B,440]
            loss_lsd = compute_avg_lsd(H_target, H_recon)
            total_loss = loss_recon + current_lsd_weight * loss_lsd

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += total_loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        avg_epoch_mse = epoch_mse_total / batch_count

        model.eval()
        val_loss_total = 0.0
        total_lsd_val = 0.0
        rot_lsd_val = 0.0
        grad_e_val = 0.0
        batches = 0

        with torch.no_grad():
            for ear_a, head_a, _, sh, subj, freq, dom in val_loader:
                if dom.dim() > 1:
                    dom_idx = dom.argmax(dim=1).long()
                else:
                    dom_idx = dom.long()
        # Reduce any extra dims
                if sh.dim() == 3:
                    sh = sh.mean(dim=-1)
                elif sh.dim() == 4:
                    sh = sh.squeeze(1).mean(dim=-1)

        # Move to device
                ear_a     = ear_a.to(device)
                head_a    = head_a.to(device)
                sh        = sh.to(device)
                freq      = freq.to(device)
                dom       = dom.to(device)
                subj      = subj.long().to(device).squeeze()



        # Forward + recon loss
                recon = model(sh, head_a, ear_a, freq, dom_idx)
                loss_recon, _ = deterministic_loss(recon, sh, recon_weight)
                sh_lin   = torch.pow(10.0, sh   / 20.0)
                recon_lin= torch.pow(10.0, recon/ 20.0)

        # Denormalize and compute LSD
                sh_lin    = torch.pow(10.0, sh    / 20.0)  # [B,64]
                recon_lin = torch.pow(10.0, recon / 20.0)  # [B,64]
                H_target  = sh_lin    @ Y.T               # [B,440]
                H_recon   = recon_lin @ Y.T               # [B,440]
                lsd      = compute_avg_lsd(H_target, H_recon)
                rot_lsd = rotationally_averaged_lsd(
                    sh_lin, recon_lin,
                    L=7, K=5,
                    measured_pts_path=os.path.join(args.measured_sh_dir, 'Measured_pts.mat')
                    ).item()
                grad_e  = directional_grad_energy(H_recon).item()

        # Total loss
                loss_total = loss_recon + current_lsd_weight * lsd

                val_loss_total += loss_total.item()
                total_lsd_val += lsd.item()
                rot_lsd_val    += rot_lsd
                grad_e_val     += grad_e
                batches += 1

# Averages
        avg_val_loss = val_loss_total / len(val_loader)
        avg_lsd      = total_lsd_val   / batches
        avg_rot_lsd = rot_lsd_val  / batches
        avg_grad_e  = grad_e_val   / batches    

# Log & print
        log_dict = {
        "epoch":       epoch + 1,
        "train_loss":  avg_train_loss,
        "val_loss":    avg_val_loss,
        "mean_LSD":         avg_lsd,
        "mean_Rot_LSD":     avg_rot_lsd,
        "mean_Grad_E":      avg_grad_e,
        "Recon Loss":  avg_epoch_mse,
        "Learning Rate": scheduler.get_last_lr()[0]
        }
        print(f"Epoch {epoch+1}: val_loss={avg_val_loss:.4f}, LSD={avg_lsd:.4f}, Recon={avg_epoch_mse:.4f}, Rot_LSD={avg_rot_lsd:.4f}, Grad_E={avg_grad_e:.4f}")
        wandb.log(log_dict)

        # best‐LSD checkpoint
        # best‑LSD checkpoint
        if avg_lsd < best_ms_lsd:
            best_ms_lsd = avg_lsd
        # per‐fold save (optional)
            torch.save(model.state_dict(),
                    os.path.join(args.model_dir, f'best_lsd_fold{args.val_idx}.pth'))
        # first‐fold canonical LSD for analysis
            if args.val_idx == 0:
                torch.save(model.state_dict(),
                       os.path.join(args.model_dir, 'canonical_lsd.pth'))
            print(f"✅ Fold {args.val_idx} best LSD: {best_ms_lsd:.4f} dB")

    # best‑Rotational‑LSD checkpoint
        if avg_rot_lsd < best_rot_lsd:
            best_rot_lsd = avg_rot_lsd
            torch.save(model.state_dict(),
                    os.path.join(args.model_dir, f'best_rot_lsd_fold{args.val_idx}.pth'))
            if args.val_idx == 0:
                torch.save(model.state_dict(),
                       os.path.join(args.model_dir, 'rotational_lsd.pth'))
            print(f"✅ Fold {args.val_idx} best Rot_LSD: {best_rot_lsd:.4f} dB")

    # best‑Directional‑Grad‐Energy checkpoint
        if avg_grad_e < best_grad_e:
            best_grad_e = avg_grad_e
            torch.save(model.state_dict(),
                    os.path.join(args.model_dir, f'best_grad_fold{args.val_idx}.pth'))
            if args.val_idx == 0:
                torch.save(model.state_dict(),
                        os.path.join(args.model_dir, 'directional_gradients.pth'))
            print(f"✅ Fold {args.val_idx} best Grad_E: {best_grad_e:.4f}")

        # Early‑stop on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(),
           os.path.join(args.model_dir, f'best_hrtf_cnn_fold{args.val_idx}.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping due to no improvement on validation loss.")
                break

    torch.save(
        model.state_dict(),
        os.path.join(args.model_dir, f'hrtf_final_fold{args.val_idx}.pth')
    )
    wandb.finish()
    return {
    'LSD':     best_ms_lsd,
    'Rot_LSD': best_rot_lsd,
    'Grad_E':  best_grad_e
}


if __name__ == "__main__":
    class Args:
        anthro_mat_path = '/Users/anishnair/Global_HRTF_VAE/Normalized_Anthropometric_Data.csv'
        measured_hrtf_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
        measured_sh_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/'
        simulated_hrtf_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Simulated/'
        simulated_sh_dir = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Simulated/'
        shvec_path     = '/Users/anishnair/Global_HRTF_VAE/Processed_ML/Measured/Measured_Ynm.mat'
        val_idx = 0
        batch_size = 1024  # Adjusted batch size based on available data/memory.
        num_freq_bins = 512
        num_workers = 0
        model_dir = "./models"

    args = Args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    # figure out how many subjects we have
    import pandas as pd
    N = pd.read_csv(args.anthro_mat_path).shape[0]

    all_results = []
    for val_idx in range(N):
        print(f"\n=== leaving out subject {val_idx} ===")
        args.val_idx = val_idx
        res = train_model(
            args,
            num_epochs=1000,
            lr=0.0005,
            patience=20,
            lsd_warmup_epochs=0.0,
            lsd_weight=0.0,
            recon_weight=1.0
        )
        all_results.append(res)

    # average across all held‑out folds
    import numpy as np
    mean_lsd    = np.mean([r['LSD']     for r in all_results])
    mean_rot    = np.mean([r['Rot_LSD'] for r in all_results])
    mean_grad   = np.mean([r['Grad_E']  for r in all_results])

    print("\n=== Leave‑one‑out summary ===")
    print(f"Mean val LSD:        {mean_lsd:.4f} dB")
    print(f"Mean val Rot_LSD:    {mean_rot:.4f} dB")
    print(f"Mean val Grad_E:     {mean_grad:.4f}")