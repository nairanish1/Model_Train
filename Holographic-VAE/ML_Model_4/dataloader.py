#### importing necessary libraries ####
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import glob
import scipy.signal  # In case you later want to resample instead of tile
from config import Args
import random
#############################################
####### HUTUBS Dataset Class (Measured Only) ######
#############################################
class HUTUBS_Dataset(Dataset):
    def __init__(self, args, val=False, include_simulated=False):
        """
        args: A namespace with keys:
              - anthro_mat_path: path to the CSV file with normalized anthropometric data.
              - measured_hrtf_dir: directory containing individual measured HRTF files 
                                   (each with shape [fft_length x 2 x num_freq_bins x num_subjects]).
              - measured_sh_dir: directory containing individual measured SH files 
                                   (each with shape [num_coeffs x fft_length, num_subjects] or [num_coeffs, fft_length, num_subjects],
                                    where num_coeffs = (L+1)^2, expected to be 64).
              - simulated_hrtf_dir: directory containing individual simulated HRTF files.
              - simulated_sh_dir: directory containing individual simulated SH files.
              - val_idx: index (integer) of the subject to use for validation.
              - batch_size: batch size for the DataLoader.
              - num_workers: number of workers for the DataLoader.
        val: Boolean flag indicating whether to load the validation subset.
        include_simulated: Boolean flag indicating whether to include simulated data.
        
        When include_simulated=False (default), only the measured HRTF and measured SH coefficients are used.
        For measured data, the shapes are:
              measured_hrtf -> [fft_length, 2, num_freq_bins, num_subjects]
              measured_sh   -> [num_coeffs, fft_length, num_subjects] (expected num_coeffs = 64)
        """
        super(HUTUBS_Dataset, self).__init__()
        self.args = args
        self.val = val
        self.include_simulated = include_simulated

        # Load anthropometric data (CSV already normalized and with subject IDs removed)
        anthro = pd.read_csv(self.args.anthro_mat_path).values.astype(np.float32)
        if self.val:
            self.anthro_mat = anthro[[self.args.val_idx], :]
        else:
            self.anthro_mat = np.delete(anthro, self.args.val_idx, axis=0)
        
        # Assume columns are ordered as:
        # first 13 columns: head measurements,
        # next 12: left ear measurements,
        # following 12: right ear measurements.
        self.anthro_head = self.anthro_mat[:, :13]
        self.anthro_left = self.anthro_mat[:, 13:25]
        self.anthro_right = self.anthro_mat[:, 25:]
        
        # Load measured data.
        measured_hrtf_files = sorted(glob.glob(os.path.join(self.args.measured_hrtf_dir, '*_HRTF_measured_dB.mat')))
        measured_sh_files   = sorted(glob.glob(os.path.join(self.args.measured_sh_dir, '*_SH_measured.mat')))
        
        print("Loading measured HRTF files...")
        measured_hrtf_list = [sio.loadmat(f)['hrtf_measured_dB'].astype(np.float32) for f in measured_hrtf_files]
        
        print("Loading measured SH files...")
        measured_sh_list = []
        for f in measured_sh_files:
            sh_data = sio.loadmat(f)['sh_coeffs_measured'].astype(np.float32)
            if sh_data.ndim == 2:
                # Expected shape: [num_coeffs, num_subjects]
                if sh_data.shape[0] != 64:
                    print(f"Warning: SH data from {f} has shape {sh_data.shape}, slicing to first 64 rows.")
                    sh_data = sh_data[:64, :]
                # Expand along frequency dimension to match FFT length.
                sh_data = sh_data[:, :, np.newaxis]
            elif sh_data.ndim == 3:
                # Expected shape: [num_coeffs, fft_length, num_subjects]
                if sh_data.shape[0] != 64:
                    print(f"Warning: SH data from {f} has shape {sh_data.shape}, slicing to first 64 coefficients.")
                    sh_data = sh_data[:64, :, :]
            else:
                raise ValueError(f"Unexpected SH data shape from {f}: {sh_data.shape}")
            measured_sh_list.append(sh_data)

        # Optionally load simulated data if include_simulated is True.
        if self.include_simulated:
            simulated_hrtf_files = sorted(glob.glob(os.path.join(self.args.simulated_hrtf_dir, '*_HRTF_simulated.mat')))
            simulated_sh_files   = sorted(glob.glob(os.path.join(self.args.simulated_sh_dir, '*_SH_simulated.mat')))
            print("Loading simulated HRTF files...")
            simulated_hrtf_list = [sio.loadmat(f)['hrtf_simulated_dB'].astype(np.float32) for f in simulated_hrtf_files]
            print("Loading simulated SH files...")
            simulated_sh_list = []
            for f in simulated_sh_files:
                sh_data = sio.loadmat(f)['sh_coeffs_simulated'].astype(np.float32)
                if sh_data.ndim == 2:
                    if sh_data.shape[0] != 64:
                        print(f"Warning: Simulated SH data from {f} has shape {sh_data.shape}, slicing to first 64 rows.")
                        sh_data = sh_data[:64, :]
                    sh_data = np.tile(sh_data[:, np.newaxis, :], (1, self.args.num_freq_bins, 1))
                elif sh_data.ndim == 3:
                    if sh_data.shape[0] != 64:
                        print(f"Warning: Simulated SH data from {f} has shape {sh_data.shape}, slicing to first 64 coefficients.")
                        sh_data = sh_data[:64, :, :]
                else:
                    raise ValueError(f"Unexpected simulated SH data shape from {f}: {sh_data.shape}")
                simulated_sh_list.append(sh_data)
        
        # Stack measured data along a new subject axis.
        measured_hrtf_all = np.stack(measured_hrtf_list, axis=-1)   # [fft,2,dirs,subjects]
        measured_sh_all   = np.concatenate(measured_sh_list, axis=2)     # shape: [64, fft_length, num_subjects]

        # If including simulated data, stack them and then concatenate along the subject/domain axis.
        if self.include_simulated:
            simulated_hrtf_all = np.stack(simulated_hrtf_list, axis=-1)
            simulated_sh_all   = np.concatenate(simulated_sh_list, axis=2)
            # Concatenate measured and simulated along the subject/domain axis.
            self.hr_tf = np.concatenate((measured_hrtf_all, simulated_hrtf_all), axis=-1)
            self.sh = np.concatenate((measured_sh_all, simulated_sh_all), axis=-1)
        else:
            self.hr_tf = measured_hrtf_all
            self.sh = measured_sh_all

        # If validation, select only the subject indicated by val_idx.
        if self.val:
            self.hr_tf = np.expand_dims(self.hr_tf[..., self.args.val_idx], axis=-1)
            self.sh = np.expand_dims(self.sh[..., self.args.val_idx], axis=-1)
        else:
            self.hr_tf = np.delete(self.hr_tf, self.args.val_idx, axis=-1)
            self.sh = np.delete(self.sh, self.args.val_idx, axis=-1)
        
        # Set the number of subjects and fft_length.
        self.fft_length = self.hr_tf.shape[0]
        self.num_subjects = self.anthro_head.shape[0]
        self.num_freq_bins = self.hr_tf.shape[0]  # axis-0: frequency bins
        self.num_dirs = self.hr_tf.shape[2]       # axis-2: directions
        # Note: if include_simulated is False, hr_tf has shape [fft_length, 2, num_freq_bins, num_subjects]
        #       if True, hr_tf has shape [fft_length, 2, num_freq_bins, 2*num_subjects]

    def __len__(self):
        domains = 4 if self.include_simulated else 2
        return self.num_subjects * self.num_freq_bins * domains
        

    def __getitem__(self, idx):
        num_subjects = self.num_subjects
        num_freq_bins = self.num_freq_bins
        domain = idx // (num_subjects * num_freq_bins)
        new_idx = idx % (num_subjects * num_freq_bins)
        freq = new_idx // num_subjects
        subject = new_idx % num_subjects

        if not self.include_simulated:
            if domain == 0:
                domain_label = np.array([1, 0, 0, 0], dtype=np.float32)  # measured left
                ear = 0
                ear_anthro = self.anthro_left[subject]
            elif domain == 1:
                domain_label = np.array([0, 1, 0, 0], dtype=np.float32)  # measured right
                ear = 1
                ear_anthro = self.anthro_right[subject]
            else:
                raise ValueError("Domain index out of range for measured data only.")

            # Corrected indexing:
            hrtf = self.hr_tf[freq, ear, :, subject]  # axis-0=freq, axis-2=dirs
            sh = self.sh[:, freq, subject]            # [num_coeffs, freq, subject]
        else:
            # ...existing code for simulated data, update indexing similarly if needed...
            pass

        head_anthro = self.anthro_head[subject]
        ear_anthro = torch.tensor(ear_anthro, dtype=torch.float32)
        head_anthro = torch.tensor(head_anthro, dtype=torch.float32)
        hrtf = torch.tensor(hrtf, dtype=torch.float32)
        sh = torch.tensor(sh, dtype=torch.float32)
        domain_label = torch.tensor(domain_label, dtype=torch.float32)

        if idx == 0:
            print("SH coefficients stats -- min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
                sh.min().item(), sh.max().item(), sh.mean().item(), sh.std().item()
            ))

        return ear_anthro, head_anthro, hrtf, sh, subject, freq, domain_label

    def train_test_split(self):
        if self.val:
            return {
                'val': (
                    (self.anthro_head, self.anthro_left, self.anthro_right),
                    self.hr_tf,
                    self.sh
                )
            }
        else:
            return {
                'train': (
                    (self.anthro_head, self.anthro_left, self.anthro_right),
                    self.hr_tf,
                    self.sh
                )
            }

#############################################
####### Dataloader Creation ######
def create_dataloader(args, seed=42):
    train_dataset = HUTUBS_Dataset(args, val=False, include_simulated=False)
    val_dataset = HUTUBS_Dataset(args, val=True, include_simulated=False)

    torch.save(train_dataset, 'train_dataset.pt', pickle_protocol=4)
    torch.save(val_dataset, 'val_dataset.pt', pickle_protocol=4)

    train_dataset = torch.load('train_dataset.pt', weights_only=False)
    val_dataset = torch.load('val_dataset.pt', weights_only=False)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    return train_loader, val_loader

def print_shape(dataset):
    dataset.measured_hrtf.shape

if __name__ == "__main__":
    # Build an args instance (adjust paths as needed)
    args = Args()
    args.device = torch.device("cpu")
    # Instantiate dataset (measured‑only)
    ds = HUTUBS_Dataset(args, val=False, include_simulated=False)
    # Pull out the very first sample
    ear_a, head_a, hrtf, sh, subj, freq, dom = ds[0]
    # hrtf should be in dB: print its statistics
    print(f"HRTF (dB) stats for subject={subj}, freq={freq}:")
    print(f"  min = {hrtf.min().item():.2f} dB")
    print(f"  max = {hrtf.max().item():.2f} dB")
    print(f"  mean = {hrtf.mean().item():.2f} dB")
    # Quick range check
    if (hrtf.max() - hrtf.min()) < 200:
        print("✅ Looks like your HRTF is in a plausible dB range.")
    else:
        print("⚠️ The range seems too large for dB—check your preprocessing!")