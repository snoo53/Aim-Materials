"""
Structure generation with composition control and geometry sanity checks.
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure

from aim_models.e3_multi_modal import AimMultiModalModel

# Full periodic table used by the model.
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",
]

AMU_TO_DENSITY_FACTOR = 1.66053906660  # rho[g/cc] = mass_amu * 1.66054 / volume_A3
ATOMIC_MASSES_AMU = []
for sym in ELEMENTS:
    try:
        ATOMIC_MASSES_AMU.append(float(Element(sym).atomic_mass))
    except Exception:
        # Fallback should never happen for the current element list.
        ATOMIC_MASSES_AMU.append(float(Element(sym).Z))

NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
RADIOACTIVE_ELEMENTS = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U"}


def build_forbidden_species_indices(forbid_noble_gas=True, forbid_radioactive=True):
    forbidden = set()
    if forbid_noble_gas:
        forbidden.update(NOBLE_GASES)
    if forbid_radioactive:
        forbidden.update(RADIOACTIVE_ELEMENTS)
    return [i for i, sym in enumerate(ELEMENTS) if sym in forbidden]


def composition_neutrality_guess(structure, max_sites=8):
    try:
        guesses = structure.composition.reduced_composition.oxi_state_guesses(max_sites=max_sites)
        return bool(len(guesses) > 0)
    except Exception:
        return None


def coordination_metrics(structure, cn_cutoff=3.0):
    n = int(len(structure))
    if n <= 0:
        return 0.0, 0
    cn_list = [len(structure.get_neighbors(structure[i], float(cn_cutoff))) for i in range(n)]
    cn_mean = float(np.mean(cn_list)) if cn_list else 0.0
    cn_max = int(np.max(cn_list)) if cn_list else 0
    return cn_mean, cn_max


def structure_plausibility_checks(
    structure,
    min_distance,
    cn_cutoff,
    cn_mean_min,
    cn_mean_max,
    cn_max_allowed,
    vpa_min,
    vpa_max,
    require_neutrality_guess,
):
    dm = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(dm, np.inf)
    min_dist = float(np.min(dm)) if dm.size else float("inf")
    cn_mean, cn_max = coordination_metrics(structure, cn_cutoff=cn_cutoff)
    neutrality = composition_neutrality_guess(structure)
    vpa = float(structure.volume) / max(int(len(structure)), 1)

    pass_min_distance = bool(np.isfinite(min_dist) and min_dist >= float(min_distance))
    pass_cn = bool(cn_mean >= float(cn_mean_min) and cn_mean <= float(cn_mean_max) and cn_max <= int(cn_max_allowed))
    pass_vpa = bool(vpa >= float(vpa_min) and vpa <= float(vpa_max))
    if require_neutrality_guess:
        pass_neutrality = bool(neutrality is True)
    else:
        pass_neutrality = True

    score = int(pass_min_distance) + int(pass_cn) + int(pass_vpa) + int(pass_neutrality)
    return {
        "min_distance": min_dist,
        "volume_per_atom": float(vpa),
        "cn_mean_r3": float(cn_mean),
        "cn_max_r3": int(cn_max),
        "composition_neutrality_guess": neutrality,
        "pass_min_distance": pass_min_distance,
        "pass_cn_reasonable": pass_cn,
        "pass_vpa_range": pass_vpa,
        "pass_neutrality_guess": pass_neutrality,
        "pass_structure_checks": bool(pass_min_distance and pass_cn and pass_vpa and pass_neutrality),
        "structure_score": int(score),
    }


def species_mass_amu(species_idx):
    idx = species_idx.long().clamp(0, len(ATOMIC_MASSES_AMU) - 1).detach().cpu().tolist()
    return float(sum(ATOMIC_MASSES_AMU[int(i)] for i in idx))


def density_from_mass_and_volume(mass_amu, volume_a3):
    if volume_a3 <= 0.0:
        return float("nan")
    return float(mass_amu) * AMU_TO_DENSITY_FACTOR / float(volume_a3)


def _quantile_clip(values, q_low=0.05, q_high=0.95):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    return arr[(arr >= lo) & (arr <= hi)].tolist()


def load_density_buckets(path, dmin, dmax, q_low=0.05, q_high=0.95):
    """
    Load realistic density distributions keyed by nelements from MP summary JSON.
    """
    out = {}
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return out

    global_vals = []
    raw = {}
    for r in rows:
        ne = r.get("nelements", None)
        den = r.get("density", None)
        try:
            ne = int(ne)
            den = float(den)
        except Exception:
            continue
        if not np.isfinite(den):
            continue
        if den < dmin or den > dmax:
            continue
        global_vals.append(den)
        raw.setdefault(ne, []).append(den)

    global_vals = _quantile_clip(global_vals, q_low=q_low, q_high=q_high)
    if global_vals:
        out["__global__"] = global_vals

    for ne, vals in raw.items():
        vals = _quantile_clip(vals, q_low=q_low, q_high=q_high)
        if vals:
            out[int(ne)] = vals
    return out


def sample_density(target_nelements, density_buckets, rng, default_density=6.0):
    if not density_buckets:
        return float(default_density)
    vals = density_buckets.get(int(target_nelements), None)
    if not vals:
        vals = density_buckets.get("__global__", None)
    if not vals:
        return float(default_density)
    return float(vals[rng.integers(0, len(vals))])


def load_natoms_buckets(path, nmin=8, nmax=64):
    """
    Load training-set num_nodes distributions keyed by nelements.
    """
    out = {}
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return out

    raw = {}
    global_vals = []
    for r in rows:
        ne = r.get("nelements", r.get("target_num_elements", None))
        nn = r.get("num_nodes", None)
        if nn is None and isinstance(r.get("node_features"), list):
            nn = len(r["node_features"])
        try:
            ne = int(ne)
            nn = int(nn)
        except Exception:
            continue
        if nn < nmin or nn > nmax:
            continue
        raw.setdefault(ne, []).append(nn)
        global_vals.append(nn)

    if global_vals:
        out["__global__"] = global_vals
    for ne, vals in raw.items():
        if vals:
            out[int(ne)] = vals
    return out


def sample_natoms(target_nelements, natoms_buckets, rng, fallback=64, nmin=8, nmax=64):
    vals = natoms_buckets.get(int(target_nelements), None) if natoms_buckets else None
    if not vals and natoms_buckets:
        vals = natoms_buckets.get("__global__", None)
    if vals:
        x = int(vals[rng.integers(0, len(vals))])
    else:
        x = int(fallback)
    x = max(int(nmin), min(int(nmax), int(x)))
    return x


def lat6_to_matrix(lat6):
    """
    Convert latent lattice-6 output to a 3x3 lattice matrix.
    Uses the same transform as training-time model code.
    """
    a, b, c, alpha, beta, gamma = [lat6[:, i] for i in range(6)]
    a = torch.clamp(a, 1e-2, 50.0)
    b = torch.clamp(b, 1e-2, 50.0)
    c = torch.clamp(c, 1e-2, 50.0)
    alpha = torch.clamp(alpha, 5.0, 175.0)
    beta = torch.clamp(beta, 5.0, 175.0)
    gamma = torch.clamp(gamma, 5.0, 175.0)

    alpha = torch.deg2rad(alpha)
    beta = torch.deg2rad(beta)
    gamma = torch.deg2rad(gamma)

    va = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    vb = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)], dim=-1)

    cx = c * torch.cos(beta)
    denom = torch.sin(gamma) + 1e-6
    cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / denom
    cz_sq = torch.clamp(c**2 - cx**2 - cy**2, min=1e-6)
    cz = torch.sqrt(cz_sq)
    vc = torch.stack([cx, cy, cz], dim=-1)

    mat = torch.stack([va, vb, vc], dim=1)
    return torch.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def pbc_min_distance(frac_coords, lattice_3x3):
    """Minimum pairwise distance using minimum-image periodic convention."""
    n = frac_coords.size(0)
    if n < 2:
        return torch.tensor(float("inf"), device=frac_coords.device)

    diff_frac = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
    diff_frac = diff_frac - torch.round(diff_frac)
    diff_cart = diff_frac @ lattice_3x3

    dist = torch.linalg.norm(diff_cart + 1e-12, dim=-1)
    dist = dist + torch.eye(n, device=dist.device) * 1e6
    return torch.min(dist)


def make_grid_fractional(n_atoms, device, dtype):
    """
    Deterministic fallback coordinates on a cubic grid (with tiny noise)
    to avoid collapsed sites.
    """
    g = int(torch.ceil(torch.tensor(float(n_atoms) ** (1.0 / 3.0))).item())
    pts = []
    for i in range(g):
        for j in range(g):
            for k in range(g):
                pts.append([(i + 0.5) / g, (j + 0.5) / g, (k + 0.5) / g])
    frac = torch.tensor(pts[:n_atoms], device=device, dtype=dtype)
    frac = torch.remainder(frac + 0.01 * torch.randn_like(frac) / max(g, 1), 1.0)
    return frac


def repair_geometry(
    frac_coords,
    lattice_3x3,
    min_distance=1.2,
    min_volume=100.0,
    max_volume=5000.0,
    mass_amu=None,
    target_density=None,
    density_min=None,
    density_max=None,
    jitter_trials=24,
):
    """
    Repair decoded structures to satisfy basic physical plausibility:
    - finite, positive-volume lattice
    - no severe overlaps
    - minimum volume floor
    """
    frac = torch.remainder(frac_coords, 1.0)
    lat = lattice_3x3.clone()
    n_atoms = max(int(frac.size(0)), 1)
    # For fixed-atom generation, enforce a practical lower bound on volume.
    effective_min_volume = max(float(min_volume), float(n_atoms) * (float(min_distance) ** 3) * 1.6)

    if not torch.isfinite(lat).all():
        side = effective_min_volume ** (1.0 / 3.0)
        lat = torch.eye(3, device=frac.device, dtype=frac.dtype) * side

    vol = torch.abs(torch.det(lat))
    if (not torch.isfinite(vol)) or vol.item() <= 1e-8:
        side = effective_min_volume ** (1.0 / 3.0)
        lat = torch.eye(3, device=frac.device, dtype=frac.dtype) * side
        vol = torch.abs(torch.det(lat))

    # Break exact duplicates first; scaling cannot fix zero-distance overlaps.
    dmin = pbc_min_distance(frac, lat)
    for t in range(jitter_trials):
        if torch.isfinite(dmin) and dmin.item() > 1e-3:
            break
        sigma = 0.05 * (0.7 ** t)
        frac = torch.remainder(frac + sigma * torch.randn_like(frac), 1.0)
        dmin = pbc_min_distance(frac, lat)

    if (not torch.isfinite(dmin)) or dmin.item() <= 1e-8:
        # Last-resort reset when decoder collapses many atoms to same point.
        frac = torch.rand_like(frac)
        dmin = pbc_min_distance(frac, lat)

    if not torch.isfinite(dmin):
        return frac, lat, False, float("nan"), float(vol.item()), float("nan")

    # If still highly collapsed, replace by packed grid fallback.
    if dmin.item() < 0.3 * min_distance:
        frac = make_grid_fractional(n_atoms, frac.device, frac.dtype)
        dmin = pbc_min_distance(frac, lat)

    # Isotropic lattice scaling for min-distance target.
    if dmin.item() < min_distance:
        scale = (min_distance / max(dmin.item(), 1e-6)) * 1.02
        scale = min(scale, 4.0)  # avoid runaway volume from pathological cases
        lat = lat * scale

    # Volume floor for dense fixed-atom cells.
    vol = torch.abs(torch.det(lat))
    if (not torch.isfinite(vol)) or vol.item() <= 1e-8:
        return frac, lat, False, float(dmin.item()), float("nan"), float("nan")
    if vol.item() < effective_min_volume:
        lat = lat * ((effective_min_volume / vol.item()) ** (1.0 / 3.0))

    # Final overlap rescue via small random perturbations.
    dmin = pbc_min_distance(frac, lat)
    if torch.isfinite(dmin) and dmin.item() < min_distance:
        for _ in range(jitter_trials):
            trial = torch.remainder(frac + 0.01 * torch.randn_like(frac), 1.0)
            trial_d = pbc_min_distance(trial, lat)
            if trial_d.item() > dmin.item():
                frac = trial
                dmin = trial_d
            if dmin.item() >= min_distance:
                break

    # One last fallback + deterministic scale if still below min-distance.
    if torch.isfinite(dmin) and dmin.item() < min_distance:
        frac = make_grid_fractional(n_atoms, frac.device, frac.dtype)
        dmin = pbc_min_distance(frac, lat)
    if torch.isfinite(dmin) and dmin.item() < min_distance:
        lat = lat * ((min_distance / max(dmin.item(), 1e-6)) * 1.01)
        dmin = pbc_min_distance(frac, lat)

    vol = torch.abs(torch.det(lat))
    density = float("nan")
    if mass_amu is not None and target_density is not None and target_density > 0.0:
        desired_vol = float(mass_amu) * AMU_TO_DENSITY_FACTOR / float(target_density)
        desired_vol = max(desired_vol, effective_min_volume)
        desired_vol = min(desired_vol, float(max_volume))
        if torch.isfinite(vol) and vol.item() > 1e-8:
            lat = lat * ((desired_vol / vol.item()) ** (1.0 / 3.0))
            dmin = pbc_min_distance(frac, lat)
            if torch.isfinite(dmin) and dmin.item() < min_distance:
                lat = lat * ((min_distance / max(dmin.item(), 1e-6)) * 1.01)
                dmin = pbc_min_distance(frac, lat)
            vol = torch.abs(torch.det(lat))

    if mass_amu is not None and torch.isfinite(vol):
        density = density_from_mass_and_volume(float(mass_amu), float(vol.item()))

    density_ok = True
    if density_min is not None and density_max is not None and np.isfinite(density):
        density_ok = (float(density_min) <= float(density) <= float(density_max))

    is_valid = (
        torch.isfinite(dmin)
        and torch.isfinite(vol)
        and dmin.item() >= min_distance
        and vol.item() >= effective_min_volume
        and vol.item() <= float(max_volume)
        and density_ok
    )
    return frac, lat, bool(is_valid), float(dmin.item()), float(vol.item()), float(density)


def decoded_to_structure(lattice_3x3, frac_coords, species_idx):
    """Convert decoded values to pymatgen Structure."""
    lat = Lattice(lattice_3x3.detach().cpu().numpy())
    idx = species_idx.detach().cpu().long().clamp(0, len(ELEMENTS) - 1)
    sp = [ELEMENTS[int(i)] for i in idx.tolist()]
    fc = torch.remainder(frac_coords, 1.0).detach().cpu().numpy()
    return Structure(lat, sp, fc, coords_are_cartesian=False)


def build_graph(frac_coords, lattice_3x3, species_idx, cutoff=5.0, max_neighbors=32):
    """Build graph representation using periodic minimum-image distances."""
    n_species = len(ELEMENTS)
    n_atoms = frac_coords.size(0)

    x = torch.zeros((n_atoms, n_species), dtype=torch.float32)
    idx = species_idx.long().clamp(0, n_species - 1)
    x[torch.arange(n_atoms), idx] = 1.0

    diff_frac = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
    diff_frac = diff_frac - torch.round(diff_frac)
    diff_cart = diff_frac @ lattice_3x3
    dist = torch.linalg.norm(diff_cart + 1e-12, dim=-1)

    edge_i, edge_j, edge_attr = [], [], []
    for i in range(n_atoms):
        d = dist[i]
        nn = torch.argsort(d)
        cnt = 0
        for j in nn.tolist():
            if i == j:
                continue
            if d[j].item() > cutoff:
                break
            edge_i.append(i)
            edge_j.append(j)
            edge_attr.append([d[j].item()])
            cnt += 1
            if cnt >= max_neighbors:
                break

    if len(edge_i) == 0:
        edge_index = [[0], [0]]
        edge_attr = [[0.0]]
    else:
        edge_index = [edge_i, edge_j]

    return {
        "node_features": x.tolist(),
        "positions": torch.remainder(frac_coords, 1.0).tolist(),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


def select_species_with_exact_count(species_logits, target_num_elements):
    """
    Restrict per-atom predictions to top-K elements and force at least one atom for each.
    species_logits: [N, n_species]
    """
    probs = torch.softmax(species_logits, dim=-1)
    scores = probs.sum(dim=0)  # [n_species]
    k = int(max(1, min(target_num_elements, scores.numel())))
    keep = torch.topk(scores, k=k).indices

    masked_logits = species_logits.clone()
    blocked = torch.ones(scores.numel(), dtype=torch.bool, device=species_logits.device)
    blocked[keep] = False
    masked_logits[:, blocked] = -1e9

    species = masked_logits.argmax(-1)
    if species.numel() >= k:
        species[:k] = keep
    return species.clamp(0, len(ELEMENTS) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    ap.add_argument("--n", type=int, default=100, help="Number of structures to generate")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--fixed_atoms",
        type=int,
        default=64,
        help="Fixed number of atoms per structure (VAE nat prediction is broken).",
    )
    ap.add_argument(
        "--natoms_mode",
        choices=["fixed", "decoded", "dataset_sample"],
        default="dataset_sample",
        help="How to choose number of atoms for each generated structure.",
    )
    ap.add_argument(
        "--natoms_dataset",
        type=str,
        default="processed_data_filtered/all_materials_data.json",
        help="Training-like JSON to sample num_nodes distribution when natoms_mode=dataset_sample.",
    )
    ap.add_argument("--natoms_min", type=int, default=8)
    ap.add_argument("--natoms_max", type=int, default=64)
    ap.add_argument(
        "--target_num_elements",
        type=int,
        default=None,
        help="Condition decode on this number of elements (e.g., 2,3,4).",
    )
    ap.add_argument(
        "--enforce_exact_num_elements",
        action="store_true",
        help="Force exactly --target_num_elements in decoded species assignments.",
    )
    ap.add_argument(
        "--min_distance",
        type=float,
        default=1.2,
        help="Minimum periodic interatomic distance in Angstrom.",
    )
    ap.add_argument(
        "--min_volume",
        type=float,
        default=100.0,
        help="Minimum cell volume in Angstrom^3.",
    )
    ap.add_argument(
        "--max_volume",
        type=float,
        default=5000.0,
        help="Maximum cell volume in Angstrom^3.",
    )
    ap.add_argument(
        "--max_attempts",
        type=int,
        default=0,
        help="Maximum candidate attempts (0 -> 20*n).",
    )
    ap.add_argument(
        "--allow_invalid",
        action="store_true",
        help="Write invalid structures too (not recommended).",
    )
    ap.add_argument(
        "--forbid_noble_gas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disallow noble gases in generated compositions.",
    )
    ap.add_argument(
        "--forbid_radioactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disallow radioactive elements in generated compositions.",
    )
    ap.add_argument(
        "--require_neutrality_guess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require oxidation-state neutrality guess to pass.",
    )
    ap.add_argument(
        "--cn_cutoff",
        type=float,
        default=3.0,
        help="Neighbor cutoff (Angstrom) for coordination sanity checks.",
    )
    ap.add_argument("--vpa_min", type=float, default=2.0, help="Minimum allowed volume-per-atom (A^3/atom).")
    ap.add_argument("--vpa_max", type=float, default=80.0, help="Maximum allowed volume-per-atom (A^3/atom).")
    ap.add_argument("--cn_mean_min", type=float, default=0.5, help="Minimum acceptable mean coordination.")
    ap.add_argument("--cn_mean_max", type=float, default=20.0, help="Maximum acceptable mean coordination.")
    ap.add_argument("--cn_max", type=int, default=32, help="Maximum acceptable per-site coordination.")
    ap.add_argument(
        "--repair_retries",
        type=int,
        default=8,
        help="Extra repair attempts per sample when structure checks fail.",
    )
    ap.add_argument(
        "--density_source_json",
        type=str,
        default="datasets/mp_summary_filtered.json",
        help="Summary JSON with density/nelements to sample target densities.",
    )
    ap.add_argument("--target_density_min", type=float, default=1.0)
    ap.add_argument("--target_density_max", type=float, default=15.0)
    ap.add_argument("--density_quantile_low", type=float, default=0.05)
    ap.add_argument("--density_quantile_high", type=float, default=0.95)
    ap.add_argument("--outdir", default="generated_cifs_fixed")
    ap.add_argument("--out_json", default="generated_materials_fixed.json")
    args = ap.parse_args()

    if args.enforce_exact_num_elements and args.target_num_elements is None:
        raise ValueError("--enforce_exact_num_elements requires --target_num_elements.")

    print("=" * 60)
    print("STRUCTURE GENERATION WITH SANITY CHECKS")
    print("=" * 60)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    print(
        f"fixed_atoms={args.fixed_atoms}, min_distance={args.min_distance}, "
        f"min_volume={args.min_volume}, max_volume={args.max_volume}"
    )
    print(
        f"natoms_mode={args.natoms_mode}, natoms_range=[{args.natoms_min},{args.natoms_max}], "
        f"density_range=[{args.target_density_min},{args.target_density_max}] g/cc"
    )
    if args.target_num_elements is not None:
        print(f"target_num_elements={args.target_num_elements}")
    if args.enforce_exact_num_elements:
        print("exact element-count enforcement: enabled")
    print(
        f"chemistry gates: forbid_noble_gas={bool(args.forbid_noble_gas)}, "
        f"forbid_radioactive={bool(args.forbid_radioactive)}, "
        f"require_neutrality_guess={bool(args.require_neutrality_guess)}"
    )
    print(
        f"coordination gates: cutoff={args.cn_cutoff:.2f} A, "
        f"cn_mean=[{args.cn_mean_min:.2f},{args.cn_mean_max:.2f}], cn_max={args.cn_max}, "
        f"vpa=[{args.vpa_min:.2f},{args.vpa_max:.2f}], repair_retries={args.repair_retries}"
    )
    if args.allow_invalid:
        print("warning: invalid structures will be kept")
    print()

    forbidden_species_indices = build_forbidden_species_indices(
        forbid_noble_gas=bool(args.forbid_noble_gas),
        forbid_radioactive=bool(args.forbid_radioactive),
    )
    forbidden_symbols = {ELEMENTS[i] for i in forbidden_species_indices}
    if forbidden_species_indices:
        print(f"forbidden_species_count={len(forbidden_species_indices)}")

    natoms_buckets = {}
    if args.natoms_mode == "dataset_sample":
        natoms_buckets = load_natoms_buckets(
            args.natoms_dataset,
            nmin=int(args.natoms_min),
            nmax=int(args.natoms_max),
        )
        print(f"loaded natoms buckets from {args.natoms_dataset}: keys={len(natoms_buckets)}")

    density_buckets = load_density_buckets(
        args.density_source_json,
        dmin=float(args.target_density_min),
        dmax=float(args.target_density_max),
        q_low=float(args.density_quantile_low),
        q_high=float(args.density_quantile_high),
    )
    print(f"loaded density buckets from {args.density_source_json}: keys={len(density_buckets)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_obj = torch.load(args.ckpt, map_location=device)

    print(f"loaded checkpoint: {args.ckpt}")
    print(f"device: {device}")

    model = AimMultiModalModel(**ckpt_obj["model_kwargs"]).to(device)
    state = ckpt_obj["model_state"]
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, val in state.items():
        if key in model_state and model_state[key].shape == val.shape:
            compatible[key] = val
        else:
            skipped.append(key)
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    if skipped:
        print(f"loaded compatible params={len(compatible)}, skipped={len(skipped)}")
    model.eval()

    os.makedirs(args.outdir, exist_ok=True)

    all_gen = []
    made = 0
    attempted = 0
    skipped_invalid = 0
    fail_geometry = 0
    fail_forbidden = 0
    fail_neutrality = 0
    fail_cn = 0
    fail_vpa = 0
    fail_write = 0
    max_attempts = args.max_attempts if args.max_attempts > 0 else max(args.n * 20, args.n)

    print(f"target_count={args.n}, max_attempts={max_attempts}")
    print("-" * 60)

    with torch.no_grad():
        while made < args.n and attempted < max_attempts:
            b = min(args.batch_size, args.n - made)

            z = args.temperature * torch.randn((b, model.vae.mu.out_features), device=device)

            target_num_elements = None
            if args.target_num_elements is not None:
                target_num_elements = torch.full((b,), int(args.target_num_elements), device=device, dtype=torch.long)

            dec = model.vae.decode(z, target_num_elements=target_num_elements)
            species_logits = dec["species_logits"]
            lattices = lat6_to_matrix(dec["lattice6"])
            max_atoms_model = int(species_logits.size(1))

            for bi in range(b):
                attempted += 1

                if args.natoms_mode == "decoded":
                    n_atoms = int(round(float(dec["nat"][bi].item())))
                elif args.natoms_mode == "dataset_sample":
                    ref_ne = int(args.target_num_elements) if args.target_num_elements is not None else 3
                    n_atoms = sample_natoms(
                        ref_ne,
                        natoms_buckets=natoms_buckets,
                        rng=rng,
                        fallback=int(args.fixed_atoms),
                        nmin=int(args.natoms_min),
                        nmax=min(int(args.natoms_max), max_atoms_model),
                    )
                else:
                    n_atoms = int(args.fixed_atoms)

                if args.target_num_elements is not None:
                    n_atoms = max(n_atoms, int(args.target_num_elements))
                n_atoms = max(int(args.natoms_min), min(int(args.natoms_max), int(n_atoms)))
                n_atoms = max(1, min(int(n_atoms), max_atoms_model))

                coords_b = dec["coords_frac"][bi, :n_atoms]
                logits_b = species_logits[bi, :n_atoms, :].clone()
                if forbidden_species_indices:
                    logits_b[:, forbidden_species_indices] = -1e9
                if args.enforce_exact_num_elements and args.target_num_elements is not None:
                    sp_b = select_species_with_exact_count(logits_b, args.target_num_elements)
                else:
                    sp_b = logits_b.argmax(-1).clamp(0, len(ELEMENTS) - 1)

                elem_counts = Counter([ELEMENTS[int(i)] for i in sp_b.cpu().tolist()])
                if forbidden_species_indices:
                    has_forbidden = any(sym in forbidden_symbols for sym in elem_counts.keys())
                    if has_forbidden and (not args.allow_invalid):
                        fail_forbidden += 1
                        skipped_invalid += 1
                        continue

                actual_ne = int(len(elem_counts))
                dens_ne = int(args.target_num_elements) if args.target_num_elements is not None else actual_ne
                target_density = sample_density(
                    dens_ne,
                    density_buckets=density_buckets,
                    rng=rng,
                    default_density=6.0,
                )
                mass_amu = species_mass_amu(sp_b)

                best = None
                n_trials = max(1, int(args.repair_retries) + 1)
                for trial in range(n_trials):
                    if trial == 0:
                        trial_coords = coords_b
                        trial_lat = lattices[bi]
                    else:
                        jitter = 0.03 * torch.randn_like(coords_b)
                        trial_coords = torch.remainder(coords_b + jitter, 1.0)
                        scale = 1.0 + 0.04 * (float(rng.random()) - 0.5)
                        trial_lat = lattices[bi] * float(scale)

                    trial_coords, trial_lat, trial_valid, trial_min_d, trial_vol, trial_rho = repair_geometry(
                        trial_coords,
                        trial_lat,
                        min_distance=float(args.min_distance),
                        min_volume=float(args.min_volume),
                        max_volume=float(args.max_volume),
                        mass_amu=float(mass_amu),
                        target_density=float(target_density),
                        density_min=float(args.target_density_min),
                        density_max=float(args.target_density_max),
                    )

                    try:
                        trial_struct = decoded_to_structure(trial_lat, trial_coords, sp_b)
                    except Exception:
                        continue

                    trial_checks = structure_plausibility_checks(
                        trial_struct,
                        min_distance=float(args.min_distance),
                        cn_cutoff=float(args.cn_cutoff),
                        cn_mean_min=float(args.cn_mean_min),
                        cn_mean_max=float(args.cn_mean_max),
                        cn_max_allowed=int(args.cn_max),
                        vpa_min=float(args.vpa_min),
                        vpa_max=float(args.vpa_max),
                        require_neutrality_guess=bool(args.require_neutrality_guess),
                    )
                    rank = int(trial_checks["structure_score"]) + int(bool(trial_valid))
                    cand = {
                        "coords": trial_coords,
                        "lat": trial_lat,
                        "is_valid": bool(trial_valid),
                        "min_d": float(trial_min_d),
                        "vol": float(trial_vol),
                        "rho": float(trial_rho),
                        "struct": trial_struct,
                        "checks": trial_checks,
                        "rank": int(rank),
                    }
                    if best is None or cand["rank"] > best["rank"]:
                        best = cand
                    if cand["is_valid"] and cand["checks"]["pass_structure_checks"]:
                        best = cand
                        break

                if best is None:
                    fail_geometry += 1
                    skipped_invalid += 1
                    continue

                coords_b = best["coords"]
                lat_b = best["lat"]
                is_valid = bool(best["is_valid"])
                min_d = float(best["min_d"])
                vol = float(best["vol"])
                rho = float(best["rho"])
                cif_struct = best["struct"]
                structure_checks = best["checks"]

                if ((not is_valid) or (not structure_checks["pass_structure_checks"])) and (not args.allow_invalid):
                    skipped_invalid += 1
                    if not is_valid:
                        fail_geometry += 1
                    if not structure_checks["pass_cn_reasonable"]:
                        fail_cn += 1
                    if not structure_checks["pass_vpa_range"]:
                        fail_vpa += 1
                    if not structure_checks["pass_neutrality_guess"]:
                        fail_neutrality += 1
                    continue

                cif_path = os.path.join(args.outdir, f"gen_{made:05d}.cif")
                try:
                    cif_struct.to(fmt="cif", filename=cif_path)
                except Exception:
                    fail_write += 1
                    skipped_invalid += 1
                    continue

                g = build_graph(coords_b.cpu(), lat_b.cpu(), sp_b.cpu())
                g["material_id"] = f"gen_{made:05d}"
                g["num_atoms"] = n_atoms
                g["min_distance"] = float(min_d)
                g["lattice_volume"] = float(vol)
                g["estimated_density"] = float(rho)
                g["target_density"] = float(target_density)
                g["lattice_matrix"] = lat_b.detach().cpu().tolist()
                g["lattice_abc"] = [float(cif_struct.lattice.a), float(cif_struct.lattice.b), float(cif_struct.lattice.c)]
                g["lattice_angles"] = [
                    float(cif_struct.lattice.alpha),
                    float(cif_struct.lattice.beta),
                    float(cif_struct.lattice.gamma),
                ]
                g["composition_neutrality_guess"] = structure_checks["composition_neutrality_guess"]
                g["cn_mean_r3"] = float(structure_checks["cn_mean_r3"])
                g["cn_max_r3"] = int(structure_checks["cn_max_r3"])
                g["pass_min_distance"] = bool(structure_checks["pass_min_distance"])
                g["pass_cn_reasonable"] = bool(structure_checks["pass_cn_reasonable"])
                g["pass_vpa_range"] = bool(structure_checks["pass_vpa_range"])
                g["pass_neutrality_guess"] = bool(structure_checks["pass_neutrality_guess"])
                g["pass_structure_checks"] = bool(structure_checks["pass_structure_checks"])
                g["volume_per_atom"] = float(structure_checks["volume_per_atom"])

                composition = "".join(
                    [f"{elem}{count}" if count > 1 else elem for elem, count in sorted(elem_counts.items())]
                )
                g["composition"] = composition
                g["nelements"] = actual_ne
                if args.target_num_elements is not None:
                    g["target_num_elements"] = int(args.target_num_elements)

                all_gen.append(g)
                made += 1

                if made == 1 or made % 10 == 0:
                    print(
                        f"[{made}/{args.n}] {composition} | n={n_atoms} | dmin={min_d:.3f} A | "
                        f"V={vol:.1f} A^3 | rho={rho:.2f} g/cc | "
                        f"vpa={structure_checks['volume_per_atom']:.2f} | "
                        f"cn={structure_checks['cn_mean_r3']:.2f}/{structure_checks['cn_max_r3']} | "
                        f"neutral={structure_checks['composition_neutrality_guess']}"
                    )

                if made >= args.n:
                    break

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_gen, f, indent=2)

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"generated={len(all_gen)}")
    print(f"attempted={attempted}, skipped_invalid={skipped_invalid}")
    print(
        "invalid_breakdown: "
        f"geometry={fail_geometry}, forbidden={fail_forbidden}, "
        f"neutrality={fail_neutrality}, cn={fail_cn}, vpa={fail_vpa}, write={fail_write}"
    )
    if len(all_gen) < args.n:
        print(f"warning: could not reach target n={args.n} within max_attempts={max_attempts}")
    print(f"cif_dir={args.outdir}")
    print(f"json={args.out_json}")

    all_elements = []
    for g in all_gen:
        elements = re.findall(r"([A-Z][a-z]?)", g.get("composition", ""))
        all_elements.extend(elements)
    elem_counts = Counter(all_elements)
    print("top elements:")
    for elem, count in elem_counts.most_common(10):
        print(f"  {elem}: {count}")


if __name__ == "__main__":
    main()
