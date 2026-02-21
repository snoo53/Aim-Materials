import torch

def lat6_to_matrix(lat6: torch.Tensor) -> torch.Tensor:
    """
    (a,b,c,alpha,beta,gamma) in degrees -> (B,3,3) lattice matrix with column vectors a,b,c
    """
    a,b,c,alpha,beta,gamma = [lat6[:,i] for i in range(6)]
    alpha = torch.deg2rad(alpha); beta = torch.deg2rad(beta); gamma = torch.deg2rad(gamma)
    va = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vb = torch.stack([b*torch.cos(gamma), b*torch.sin(gamma), torch.zeros_like(b)], dim=-1)
    cx = c*torch.cos(beta)
    cy = c*(torch.cos(alpha) - torch.cos(beta)*torch.cos(gamma)) / (torch.sin(gamma)+1e-9)
    cz = torch.sqrt((c**2 - cx**2 - cy**2).clamp_min(1e-9))
    vc = torch.stack([cx, cy, cz], dim=-1)
    L = torch.stack([va, vb, vc], dim=1)
    return L

def min_image_distances(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    Pairwise distances under PBC using minimum-image convention.
    frac: (N,3) fractional; lattice: (3,3) cartesian columns
    returns (N,N) distances
    """
    d_frac = frac.unsqueeze(1) - frac.unsqueeze(0)           # (N,N,3)
    d_frac = d_frac - d_frac.round()                         # wrap to [-0.5,0.5]
    d_cart = torch.matmul(d_frac, lattice)                   # (N,N,3)
    d = torch.linalg.norm(d_cart, dim=-1)
    return d

def soft_min_distance(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    d = min_image_distances(frac, lattice) + torch.eye(frac.size(0), device=frac.device)*1e6
    return d.min()
