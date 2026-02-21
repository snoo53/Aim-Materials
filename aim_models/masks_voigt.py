import torch

# Define index tensors at module level (outside function)
_VOIGT21_I = torch.tensor([0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5], dtype=torch.long)
_VOIGT21_J = torch.tensor([0,1,2,3,4,5,1,2,3,4,5,2,3,4,5,3,4,5,4,5,5], dtype=torch.long)

def pack_voigt_6x6_sym(C: torch.Tensor) -> torch.Tensor:
    """
    Pack a symmetric 6x6 matrix to 21-vector using row-major upper triangle (i<=j).
    Vectorized version for better performance.
    
    Args:
        C: (..., 6, 6) symmetric tensor
    
    Returns:
        (..., 21) packed vector
    """
    # Move index tensors to same device as input
    i_idx = _VOIGT21_I.to(C.device)
    j_idx = _VOIGT21_J.to(C.device)
    
    # Single advanced indexing operation (much faster than loop)
    return C[..., i_idx, j_idx]

def unpack_voigt_21_sym(v: torch.Tensor) -> torch.Tensor:
    """
    Inverse of pack_voigt_6x6_sym. v: (..., 21) -> (..., 6, 6)
    Vectorized version for better performance.
    
    Args:
        v: (..., 21) packed vector
    
    Returns:
        (..., 6, 6) symmetric tensor
    """
    C = torch.zeros(*v.shape[:-1], 6, 6, device=v.device, dtype=v.dtype)
    
    # Move index tensors to same device
    i_idx = _VOIGT21_I.to(v.device)
    j_idx = _VOIGT21_J.to(v.device)
    
    # Expand indices for batch dimensions
    # Create indices that work with arbitrary leading dimensions
    for k in range(21):
        C[..., i_idx[k], j_idx[k]] = v[..., k]
        C[..., j_idx[k], i_idx[k]] = v[..., k]  # Symmetry
    
    return C

def symmetry_mask_6x6(crystal_system: str) -> torch.Tensor:
    """
    Returns a boolean (6,6) mask of allowed nonzero entries for the elastic stiffness tensor C (Voigt).
    This mask only controls zero vs nonzero (independent-parameter equality constraints are handled by a separate loss).
    Based on standard forms (Nye, Musgrave). Crystal systems:
      triclinic (21), monoclinic (13), orthorhombic (9), tetragonal (6/7),
      trigonal (6/7), hexagonal (5), cubic (3).
    We return a *permissive* mask that matches common textbooks.

    Note: We do not distinguish the two tetragonal/trigonal settings here—use equality losses if needed.
    """
    cs = crystal_system.lower()
    M = torch.ones(6,6, dtype=torch.bool)
    # Symmetry zeros (common forms)
    if cs in ["triclinic"]:
        pass
    elif cs in ["monoclinic"]:
        # assume unique axis b (2): many C_{i4},C_{i6} zeroed except certain blocks
        Z = [(0,3),(0,5),(1,3),(1,5),(2,3),(2,5),
             (3,0),(3,1),(3,2),(3,4),(3,5),
             (4,3),(4,5),
             (5,0),(5,1),(5,2),(5,3),(5,4)]
        for i,j in Z: M[i,j]=False
    elif cs in ["orthorhombic"]:
        # zero shear-normal couplings
        Z = [(0,3),(0,4),(0,5),
             (1,3),(1,4),(1,5),
             (2,3),(2,4),(2,5),
             (3,0),(3,1),(3,2),(3,4),(3,5),
             (4,0),(4,1),(4,2),(4,3),(4,5),
             (5,0),(5,1),(5,2),(5,3),(5,4)]
        for i,j in Z: M[i,j]=False
    elif cs in ["tetragonal","trigonal","hexagonal","cubic"]:
        # progressively more zeros
        # Start from orthorhombic zeros
        Z = [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),
             (3,0),(3,1),(3,2),(3,4),(3,5),
             (4,0),(4,1),(4,2),(4,3),(4,5),
             (5,0),(5,1),(5,2),(5,3),(5,4)]
        for i,j in Z: M[i,j]=False
        if cs in ["tetragonal","trigonal","hexagonal","cubic"]:
            # enforce C16=C26=0 commonly
            for ij in [(0,5),(1,5),(5,0),(5,1)]: M[ij]=False
        if cs in ["hexagonal","trigonal","cubic"]:
            # C14=C24=0 commonly (hex/cubic)
            for ij in [(0,3),(1,3),(3,0),(3,1)]: M[ij]=False
        if cs in ["cubic"]:
            # also C12=C13, C22=C33=C11, C44=C55=C66; mask keeps only diagonal blocks
            # allow only {C11,C12,C44} families -> we mark allowed; rest false
            M = torch.zeros(6,6, dtype=torch.bool)
            for i in [0,1,2]: M[i,i]=True     # C11,C22,C33
            for (i,j) in [(0,1),(0,2),(1,2)]: M[i,j]=True; M[j,i]=True  # C12,C13,C23
            for k in [3,4,5]: M[k,k]=True     # C44,C55,C66
    return M

def symmetry_mask_21(crystal_system: str, device="cpu") -> torch.Tensor:
    M6 = symmetry_mask_6x6(crystal_system).to(device)
    v = pack_voigt_6x6_sym(M6.to(torch.float32)) > 0.5
    return v

def equality_penalties(C_pred_21: torch.Tensor, crystal_systems: list[str]) -> torch.Tensor:
    """
    Soft equality constraints (average MAE) for systems where certain entries must be equal:
      cubic: C11=C22=C33; C12=C13=C23; C44=C55=C66; and C66=(C11-C12)/2 (often listed as consequence for hex)
      hexagonal: C11=C22; C66=(C11-C12)/2
      tetragonal/trigonal: partial equalities (we include a minimal set)
    These encourage (not enforce) the relations.

    Returns scalar penalty.
    """
    if C_pred_21.numel() == 0: return torch.zeros([], device=C_pred_21.device)
    C = unpack_voigt_21_sym(C_pred_21)  # (B,6,6)
    pen = 0.0
    B = C.shape[0]
    for b in range(B):
        cs = crystal_systems[b].lower()
        Cb = C[b]
        if cs == "cubic":
            pen += (Cb[0,0]-Cb[1,1]).abs() + (Cb[0,0]-Cb[2,2]).abs()
            pen += (Cb[0,1]-Cb[0,2]).abs() + (Cb[0,1]-Cb[1,2]).abs()
            pen += (Cb[3,3]-Cb[4,4]).abs() + (Cb[3,3]-Cb[5,5]).abs()
        elif cs == "hexagonal":
            pen += (Cb[0,0]-Cb[1,1]).abs()
            pen += (Cb[5,5] - 0.5*(Cb[0,0]-Cb[0,1])).abs()
        elif cs == "tetragonal":
            # minimal equalities: C11=C22, C13=C23, C44=C55
            pen += (Cb[0,0]-Cb[1,1]).abs()
            pen += (Cb[0,2]-Cb[1,2]).abs()
            pen += (Cb[3,3]-Cb[4,4]).abs()
        elif cs == "trigonal":
            pen += (Cb[0,0]-Cb[1,1]).abs()
            pen += (Cb[3,3]-Cb[4,4]).abs()
        # orthorhombic/monoclinic/triclinic: no equalities
    return pen / max(1, B)
