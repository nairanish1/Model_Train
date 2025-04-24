# ML_Model/utils_metrics.py

import math
import torch
import torch.nn.functional as F
import scipy.io as sio
from e3nn import o3

# cache for your 440-point set
_MEASURED_PTS = None
def _load_measured_pts(path: str, device: torch.device):
    global _MEASURED_PTS
    if _MEASURED_PTS is None:
        data = sio.loadmat(path)
        pts_np = data['pts']          # your [440×3] array
        _MEASURED_PTS = torch.from_numpy(pts_np).to(device).float()
    return _MEASURED_PTS

def _make_grid_pts(L: int, device: torch.device):
    S = (L+1)**2
    N = int(math.isqrt(S))
    az = torch.linspace(0, 2*math.pi*(1-1/N), N, device=device)
    el = torch.linspace(0, math.pi, N, device=device)
    azg, elg = torch.meshgrid(az, el, indexing='ij')
    return torch.stack([
        torch.sin(elg).flatten()*torch.cos(azg).flatten(),
        torch.sin(elg).flatten()*torch.sin(azg).flatten(),
        torch.cos(elg).flatten()
    ], dim=1)  # [S,3]

def rotationally_averaged_lsd(
    sh_t: torch.Tensor,
    sh_p: torch.Tensor,
    L: int,
    K: int = 5,
    eps: float = 1e-3,
    measured_pts_path: str = None,
) -> torch.Tensor:
    """
    E_R [ 20 * mean(|log10((H_t(R)+eps)/(H_p(R)+eps))|) ] over SO(3),
    by rotating either your 440-point grid (if given) or an (L+1)^2 grid.
    """
    B, _ = sh_t.shape
    device = sh_t.device

    # choose pts
    if measured_pts_path:
        pts = _load_measured_pts(measured_pts_path, device)  # [440,3]
    else:
        pts = _make_grid_pts(L, device)                      # [S,3]

    acc = torch.zeros((), device=device, dtype=pts.dtype)

    for _ in range(K):
        # sample random R = Rz(α) · Ry(β) · Rz(γ)
        α = torch.rand(1).item() * 2*math.pi
        β = torch.rand(1).item() * math.pi
        γ = torch.rand(1).item() * 2*math.pi

        ca, sa = math.cos(α), math.sin(α)
        cb, sb = math.cos(β), math.sin(β)
        cg, sg = math.cos(γ), math.sin(γ)

        Rz1 = torch.tensor([[ ca, -sa, 0],[ sa, ca, 0],[0,0,1]], device=device, dtype=pts.dtype)
        Ry  = torch.tensor([[ cb, 0, sb],[0,1,0],[-sb,0,cb]],     device=device, dtype=pts.dtype)
        Rz2 = torch.tensor([[ cg, -sg, 0],[ sg, cg, 0],[0,0,1]],  device=device, dtype=pts.dtype)

        R = Rz1 @ Ry @ Rz2                               # [3,3]
        pts_r = pts @ R.T                                # [num_pts,3]

        # build rotated SH basis
        Y_r = torch.cat([
            o3.spherical_harmonics(l, pts_r, normalize='component').real
            for l in range(L+1)
        ], dim=1)                                       # [num_pts,(L+1)^2]

        # project
        Ht = (sh_t @ Y_r.T).abs()                       # [B, num_pts]
        Hp = (sh_p @ Y_r.T).abs()                       # [B, num_pts]

        ratio = ((Ht + eps)/(Hp + eps)).clamp(min=1e-10)
        acc += (20.0 * torch.log10(ratio).abs()).mean()

    return acc / K
# ------------------------------------------------------------------------------
def directional_grad_energy(H: torch.Tensor) -> torch.Tensor:
    """
    RMS angular‐gradient ‖∇H‖ on an equirectangular N×N grid.
    H : [B, S]   with S = N²
    returns : [scalar]
    """
    B, S = H.shape
    N    = int(math.isqrt(S))
    Hmap = H.view(B, N, N)

    # central differences
    dθ = (Hmap[:,2:,:] - Hmap[:,:-2,:]) / 2
    dφ = (Hmap[:,:,2:] - Hmap[:,:,:-2]) / 2

    # pad back to N×N by replicating the edge diffs
    dθ = F.pad(dθ, (0,0,1,1), mode='replicate')
    dφ = F.pad(dφ, (1,1,0,0), mode='replicate')

    return torch.sqrt((dθ.pow(2) + dφ.pow(2)).mean())
