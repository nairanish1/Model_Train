import torch
import torch.nn as nn

############################################
# 6′) Convolutional Encoder
############################################
class EncoderConv(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        h = x.unsqueeze(1)       # [B,1,64]
        h = self.net(h)          # [B,4,64]
        return h.mean(dim=2)     # [B,4]

############################################
# 7′) Convolutional Decoder
############################################
class DecoderConv(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4*64)
        self.net = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(4, 1, kernel_size=3, padding=1),
        )
    def forward(self, z):
        h = self.fc(z)                # [B,4*64]
        h = h.view(z.size(0),4,64)    # [B,4,64]
        return self.net(h).squeeze(1) # [B,64]

############################################
# 9′) ConvHyperCNN with FC‑fusion (~52 K params)
############################################
class ConvHyperCNN(nn.Module):
    def __init__(self, num_freq_bins, latent_dim=128, anthro_input_dim=25):
        super().__init__()
        # 1) Conv encoder → 4 features
        self.encoder_conv = EncoderConv(1)
        self.latent_proj  = nn.Linear(4, latent_dim)

        # 2) Conditioning branch
        self.anthro_encoder = nn.Sequential(
            nn.Linear(anthro_input_dim, 12), nn.ReLU(),
            nn.Linear(12, 12),
        )  # →12 dims
        self.freq_emb   = nn.Embedding(num_freq_bins, 6)  # → 6 dims
        self.domain_emb = nn.Embedding(4, 4)              # → 4 dims

        # cond_dim = 12 + 6 + 4 = 22
        cond_dim  = 12 + 6 + 4
        fused_dim = latent_dim + cond_dim               # = 120 + 22 = 142

        # 3) single‐layer fusion: 142→120
        self.fc_fuse = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # 4) Conv decoder
        self.decoder_conv = DecoderConv(latent_dim)

    def forward(self, sh_input, head_anthro, ear_anthro, freq_idx, domain_idx):
        # encode
        h = self.encoder_conv(sh_input)    # [B,4]
        z = self.latent_proj(h)            # [B,120]

        # build cond
        anthrop = torch.cat([head_anthro, ear_anthro], dim=1)
        anthro  = self.anthro_encoder(anthrop) 
        freq_idx = freq_idx.squeeze(-1).long() if freq_idx.dim()>1 else freq_idx.long()
        freq   = self.freq_emb(freq_idx)
        if domain_idx.dim()>1: domain_idx = domain_idx.argmax(dim=1)
        dom    = self.domain_emb(domain_idx.long())
        cond   = torch.cat([anthro, freq, dom], dim=1)        # [B,22]

        # fuse
        fused  = torch.cat([z, cond], dim=1)                  # [B,142]
        z      = self.fc_fuse(fused)                          # [B,120]

        # decode
        return self.decoder_conv(z)                           # [B,64]
