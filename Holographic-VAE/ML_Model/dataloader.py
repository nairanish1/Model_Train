"""
Leave-one-out loader that returns only anthropometry + (freq, ear)   ➜   SH-vector
No SH is ever fed to the network – it is the **target**.
"""

import os, glob, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

class HUTUBS_Dataset(Dataset):
    def __init__(self, args, val=False):
        super().__init__()
        self.val = val
        # ----- anthropometry -----
        anthro = pd.read_csv(args.anthro_mat_path).values.astype(np.float32)
        if val:
            anthro = anthro[[args.val_idx], :]
        else:
            anthro = np.delete(anthro, args.val_idx, axis=0)

        self.head = anthro[:, :13]          # [N,13]
        self.left = anthro[:, 13:25]        # [N,12]
        self.right= anthro[:, 25:]          # [N,12]

        # ----- SH coeffs (ground truth) -----
        files = sorted(glob.glob(os.path.join(args.measured_sh_dir, '*_SH_measured.mat')))
        sh_lst = []
        for f in files:
            sh = sio.loadmat(f)['sh_coeffs_measured'].astype(np.float32)[:64]
            sh = sh[..., None] if sh.ndim == 2 else sh      # [64, F, 1] or [64, F, S]
            sh_lst.append(sh)
        sh_all = np.concatenate(sh_lst, axis=2)             # [64, F, #subs]
        sh_all = sh_all.transpose(2,1,0)                   # [#subs, F, 64]

        self.sh = sh_all[[args.val_idx]] if val else np.delete(sh_all, args.val_idx, 0)

        self.S, self.F = self.sh.shape[0], self.sh.shape[1] # subjects, freqs

    def __len__(self):  return self.S * self.F * 2          # 2 ears / freq

    def __getitem__(self, idx):
        per_sub  = self.F * 2
        subj     = idx // per_sub
        rem      = idx %  per_sub
        ear_idx  = rem // self.F                 # 0-left  1-right
        freq_idx = rem %  self.F

        head  = torch.from_numpy(self.head[subj])
        ear   = torch.from_numpy((self.left if ear_idx==0 else self.right)[subj])
        sh    = torch.from_numpy(self.sh[subj, freq_idx])

        return head.float(), ear.float(), torch.tensor(freq_idx), torch.tensor(ear_idx), sh.float()


def create_dataloader(args, seed=42):
    g = torch.Generator().manual_seed(seed)
    train = HUTUBS_Dataset(args, val=False)
    val   = HUTUBS_Dataset(args, val=True)

    worker_fn = lambda wid: random.seed(seed + wid)

    dl_train = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, worker_init_fn=worker_fn,
                          generator=g)
    dl_val   = DataLoader(val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, worker_init_fn=worker_fn,
                          generator=g)
    return dl_train, dl_val

