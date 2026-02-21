import torch
import torch.nn.functional as F

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B,21), mask: (B,21) bool
    """
    if pred.numel() == 0: return torch.zeros([], device=pred.device)
    diff = (pred - target).abs()
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else torch.zeros([], device=pred.device)

def mae_per_column(pred: torch.Tensor, target: torch.Tensor, names=None):
    with torch.no_grad():
        mae = (pred - target).abs().mean(dim=0).cpu().numpy().tolist()
    if names is None: return mae
    return dict(zip(names, mae))
