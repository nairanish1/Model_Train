import torch
import torch.nn as nn
from typing import Dict

############################################
# 6′) Convolutional Encoder
############################################
class EncoderConv(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 1×64 → 8×64 → 16×64 → 8×64
        self.net = nn.Sequential(
            nn.Conv1d(in_channels,  4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4,             8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8,             4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,64] → [B,1,64]
        h = x.unsqueeze(1)
        h = self.net(h)       # [B, 8,64]
        return h.mean(dim=2)  # → [B, 8]

############################################
# 7′) Convolutional Decoder
############################################
class DecoderConv(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # latent → 8×64 feature map
        self.fc = nn.Linear(latent_dim, 4 * 64)
        self.net = nn.Sequential(
            nn.Conv1d(4,   8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8,   4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4,   1, kernel_size=3, padding=1),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        h = self.fc(z)                      # [B, 8*64]
        h = h.view(z.size(0), 4, 64)        # [B, 8,64]
        h = self.net(h)                     # [B, 1,64]
        return h.squeeze(1)                 # → [B,64]

############################################
# 9′) Deterministic Model: ConvHyperCNN (≈92 K params)
############################################
class ConvHyperCNN(nn.Module):
    def __init__(
        self,
        num_freq_bins: int,
        latent_dim: int = 128,
        anthro_input_dim: int = 25,
    ):
        super().__init__()
        # 1) Conv encoder
        self.encoder_conv   = EncoderConv(in_channels=1)
        self.latent_proj    = nn.Linear(4, latent_dim)

        # 2) Conditioning Branch (light MLP + small embeddings)
        self.anthro_encoder = nn.Sequential(
            nn.Linear(anthro_input_dim, 12), nn.ReLU(),
            nn.Linear(12,12),
        )
        self.freq_emb        = nn.Embedding(num_freq_bins, 10)
        self.domain_emb      = nn.Embedding(4,                4)

        cond_dim = 12 + 10 + 4  # =26

        # 3) FiLM
        self.film = nn.ModuleDict({
            "gamma": nn.Linear(cond_dim, latent_dim),
            "beta":  nn.Linear(cond_dim, latent_dim),
        })

        # 4) Conv decoder
        self.decoder_conv = DecoderConv(latent_dim)

    def forward(
        self,
        sh_input:   torch.Tensor,      # [B,64]
        head_anthro:torch.Tensor,      # [B,13]
        ear_anthro: torch.Tensor,      # [B,12]
        freq_idx:   torch.Tensor,  # [B]
        domain_idx: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        # — encode
        h = self.encoder_conv(sh_input)       # [B,8]
        z = self.latent_proj(h)               # [B,128]

        # — build cond vector
        anthro = self.anthro_encoder(torch.cat([head_anthro, ear_anthro], dim=1))

        # make sure freq_idx is a 1D LongTensor
        if freq_idx.dim() > 1:
           freq_idx = freq_idx.squeeze(-1)
        freq_idx = freq_idx.long()
        freq = self.freq_emb(freq_idx)

        # make sure domain_idx is a 1D LongTensor (if one‑hot float, argmax it)
        if domain_idx.dim() > 1:
            domain_idx = domain_idx.argmax(dim=1)
        domain_idx = domain_idx.long()
        dom = self.domain_emb(domain_idx)          
        cond   = torch.cat([anthro, freq, dom], dim=1)                # [B,48]

        # — FiLM
        γ = self.film["gamma"](cond)   # [B,128]
        β = self.film["beta"](cond)    # [B,128]
        z = z * γ + β

        # — decode
        return self.decoder_conv(z)    # [B,64]
