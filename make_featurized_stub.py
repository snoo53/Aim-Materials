"""
Featurize generated structures to match training distribution as closely as possible.

Replaces the old random-projection stub with:
- 40-d node features from element properties (same schema as processed_filtered_mp.py),
- 6-d edge features [bond_type_onehot(4), normalized_distance, normalized_en_diff],
- realistic 105-d global feature templates deterministically aggregated from training data by nelements/num_nodes.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",
]

OXI_RANGE = list(range(-3, 8))  # 11 dims
METAL_GROUPS = {
    "Actinide",
    "Alkali metal",
    "Alkaline earth metal",
    "Lanthanide",
    "Post-transition metal",
    "Transition metal",
}


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _normalize(v, mean, std):
    return (v - mean) / std if std != 0.0 else 0.0


def _encode_oxidation_states(text):
    vec = [0.0] * len(OXI_RANGE)
    if text is None:
        return vec
    parts = str(text).split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            idx = OXI_RANGE.index(int(float(p)))
            vec[idx] = 1.0
        except Exception:
            continue
    return vec


def build_element_feature_bank(elements_csv):
    df = pd.read_csv(elements_csv)
    df["StandardState"] = df["StandardState"].astype(str)
    df["GroupBlock"] = df["GroupBlock"].astype(str)

    # Numeric block used in training preprocessing.
    num_cols = [
        "AtomicMass",
        "Electronegativity",
        "AtomicRadius",
        "IonizationEnergy",
        "ElectronAffinity",
        "MeltingPoint",
        "BoilingPoint",
        "Density",
        "YoungModulus",
        "BulkModulus",
        "ShearModulus",
        "PoissonRatio",
    ]

    # Keep only columns used by preprocessing logic.
    cols = df[
        [
            "Symbol",
            "AtomicMass",
            "Electronegativity",
            "AtomicRadius",
            "IonizationEnergy",
            "ElectronAffinity",
            "OxidationStates",
            "StandardState",
            "MeltingPoint",
            "BoilingPoint",
            "Density",
            "GroupBlock",
            "YoungModulus",
            "BulkModulus",
            "ShearModulus",
            "PoissonRatio",
        ]
    ].set_index("Symbol")

    cols["has_young_modulus"] = cols["YoungModulus"].notna().astype(float)
    cols["has_bulk_modulus"] = cols["BulkModulus"].notna().astype(float)
    cols["has_shear_modulus"] = cols["ShearModulus"].notna().astype(float)
    cols["has_poisson_ratio"] = cols["PoissonRatio"].notna().astype(float)

    # Match training preprocessing: gas rows fill NaN with 0, solids use mean fill for modulus columns.
    gas_rows = cols["StandardState"].str.lower() == "gas"
    cols.loc[gas_rows] = cols.loc[gas_rows].fillna(0)
    solid_rows = cols["StandardState"].str.lower() == "solid"
    for c in ["YoungModulus", "BulkModulus", "ShearModulus", "PoissonRatio"]:
        mean_val = cols.loc[solid_rows, c].mean()
        cols.loc[solid_rows, c] = cols.loc[solid_rows, c].fillna(mean_val)

    # One-hot categories in deterministic sorted order.
    state_cats = sorted(cols["StandardState"].astype(str).unique().tolist())
    group_cats = sorted(cols["GroupBlock"].astype(str).unique().tolist())
    state_index = {k: i for i, k in enumerate(state_cats)}
    group_index = {k: i for i, k in enumerate(group_cats)}

    # Means/stds for numeric normalization.
    means = {c: _safe_float(cols[c].mean(), 0.0) for c in num_cols}
    stds = {c: _safe_float(cols[c].std(), 1.0) for c in num_cols}

    bank = {}
    electronegativity = {}
    is_metal = {}
    for sym, row in cols.iterrows():
        state_vec = [0.0] * len(state_cats)
        state = str(row["StandardState"])
        if state in state_index:
            state_vec[state_index[state]] = 1.0

        group_vec = [0.0] * len(group_cats)
        group = str(row["GroupBlock"])
        if group in group_index:
            group_vec[group_index[group]] = 1.0

        num = [_normalize(_safe_float(row[c], 0.0), means[c], stds[c]) for c in num_cols]
        oxi = _encode_oxidation_states(row.get("OxidationStates", ""))
        mod_ind = [
            _safe_float(row["has_young_modulus"], 0.0),
            _safe_float(row["has_bulk_modulus"], 0.0),
            _safe_float(row["has_shear_modulus"], 0.0),
            _safe_float(row["has_poisson_ratio"], 0.0),
        ]

        feat = num + state_vec + oxi + group_vec + mod_ind
        # Expected: 12 + 3 + 11 + 10 + 4 = 40
        if len(feat) < 40:
            feat = feat + [0.0] * (40 - len(feat))
        elif len(feat) > 40:
            feat = feat[:40]

        bank[sym] = [float(x) for x in feat]
        electronegativity[sym] = _safe_float(row["Electronegativity"], 0.0)
        is_metal[sym] = group in METAL_GROUPS

    return bank, electronegativity, is_metal


def species_symbols_from_material(m):
    nf = np.array(m.get("node_features", []), dtype=np.float32)
    if nf.ndim == 2 and nf.shape[1] >= len(ELEMENTS):
        idx = np.argmax(nf[:, : len(ELEMENTS)], axis=1).tolist()
        return [ELEMENTS[int(i)] for i in idx]

    # Fallback from composition string; used only when atom-level species is absent.
    comp = str(m.get("composition", ""))
    tokens = re_findall_formula(comp)
    symbols = []
    for sym, cnt in tokens:
        symbols.extend([sym] * cnt)
    if len(symbols) == 0:
        n = int(m.get("num_atoms", m.get("num_nodes", 64)))
        symbols = ["Si"] * n
    return symbols


def re_findall_formula(comp):
    import re

    out = []
    for sym, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", comp):
        n = int(cnt) if cnt else 1
        out.append((sym, n))
    return out


def get_edge_arrays(m):
    edge_index = m.get("edge_index", [])
    if isinstance(edge_index, list) and len(edge_index) == 2:
        src = list(edge_index[0])
        dst = list(edge_index[1])
    else:
        src, dst = [0], [0]

    edge_attr = np.array(m.get("edge_attr", []), dtype=np.float32)
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.reshape(-1, 1)
    if edge_attr.ndim != 2 or edge_attr.shape[0] != len(src):
        edge_attr = np.zeros((len(src), 1), dtype=np.float32)
    return src, dst, edge_attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--node_dim", type=int, default=40)
    ap.add_argument("--edge_dim", type=int, default=6)
    ap.add_argument("--global_dim", type=int, default=105)
    ap.add_argument("--elements_csv", default="datasets/PubChemElements_all.csv")
    ap.add_argument("--reference_data", default="processed_data_filtered/all_materials_data.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.node_dim != 40:
        raise ValueError("This featurizer is designed for node_dim=40.")
    if args.edge_dim != 6:
        raise ValueError("This featurizer is designed for edge_dim=6.")
    if args.global_dim != 105:
        raise ValueError("This featurizer is designed for global_dim=105.")

    mats = json.load(open(args.in_json, "r", encoding="utf-8"))
    ref = json.load(open(args.reference_data, "r", encoding="utf-8"))
    bank, en_lookup, metal_lookup = build_element_feature_bank(args.elements_csv)

    # Build realistic global-feature template pools.
    pool_by_ne = defaultdict(list)
    all_templates = []
    for r in ref:
        gf = r.get("global_features")
        if not isinstance(gf, list) or len(gf) < args.global_dim:
            continue
        ne = int(r.get("nelements", r.get("target_num_elements", 0)) or 0)
        nn = int(r.get("num_nodes", len(r.get("node_features", [])) or 0))
        tpl = [float(x) for x in gf[: args.global_dim]]
        pool_by_ne[ne].append((tpl, nn))
        all_templates.append((tpl, nn))
    if len(all_templates) == 0:
        all_templates = [([0.0] * args.global_dim, 64)]

    # First pass: collect raw edge-distance / EN-diff for dataset-level normalization.
    edge_raw_r = []
    edge_raw_en = []
    cached = []
    for m in mats:
        syms = species_symbols_from_material(m)
        src, dst, edge_attr = get_edge_arrays(m)
        if len(syms) == 0:
            syms = ["Si"] * max(1, len(src))
        # Distance from existing edge_attr first column if available.
        if edge_attr.shape[1] >= 1:
            rvals = edge_attr[:, 0].astype(float)
        else:
            rvals = np.zeros((len(src),), dtype=float)
        env = []
        for u, v in zip(src, dst):
            su = syms[int(u) % len(syms)]
            sv = syms[int(v) % len(syms)]
            env.append(abs(en_lookup.get(su, 0.0) - en_lookup.get(sv, 0.0)))
        env = np.array(env, dtype=float)
        edge_raw_r.extend(rvals.tolist())
        edge_raw_en.extend(env.tolist())
        cached.append((m, syms, src, dst, rvals, env))

    r_mean = float(np.mean(edge_raw_r)) if len(edge_raw_r) else 0.0
    r_std = float(np.std(edge_raw_r)) if len(edge_raw_r) else 1.0
    if r_std < 1e-8:
        r_std = 1.0
    en_mean = float(np.mean(edge_raw_en)) if len(edge_raw_en) else 0.0
    en_std = float(np.std(edge_raw_en)) if len(edge_raw_en) else 1.0
    if en_std < 1e-8:
        en_std = 1.0

    out = []
    for m, syms, src, dst, rvals, env in cached:
        n_nodes = len(syms)
        if n_nodes <= 0:
            n_nodes = int(m.get("num_atoms", m.get("num_nodes", 64)))
            syms = ["Si"] * n_nodes

        # Node features (40-d real element descriptors).
        nf2 = np.array([bank.get(sym, bank.get("Si", [0.0] * 40)) for sym in syms], dtype=np.float32)

        # Chemistry-aware edge features (6-d).
        if len(src) == 0:
            src, dst = [0], [0]
            rvals = np.array([0.0], dtype=float)
            env = np.array([0.0], dtype=float)
        ea2 = np.zeros((len(src), args.edge_dim), dtype=np.float32)
        for i, (u, v) in enumerate(zip(src, dst)):
            su = syms[int(u) % len(syms)]
            sv = syms[int(v) % len(syms)]
            en_diff = float(env[i])
            both_metal = bool(metal_lookup.get(su, False) and metal_lookup.get(sv, False))
            if both_metal and en_diff < 0.4:
                onehot = [1.0, 0.0, 0.0, 0.0]  # metallic
            elif en_diff >= 1.7:
                onehot = [0.0, 1.0, 0.0, 0.0]  # ionic
            elif en_diff >= 0.4:
                onehot = [0.0, 0.0, 1.0, 0.0]  # polar covalent
            else:
                onehot = [0.0, 0.0, 0.0, 1.0]  # nonpolar covalent
            ea2[i, :4] = np.array(onehot, dtype=np.float32)
            ea2[i, 4] = np.float32((float(rvals[i]) - r_mean) / r_std)
            ea2[i, 5] = np.float32((en_diff - en_mean) / en_std)

        # Global features: deterministic realistic template by nelements and similar node count.
        ne = int(m.get("nelements", m.get("target_num_elements", 0)) or 0)
        candidates = pool_by_ne.get(ne, None)
        if not candidates:
            candidates = all_templates
        # Aggregate top-k nearest templates by |num_nodes - template_num_nodes| for stable realism.
        cand_sorted = sorted(candidates, key=lambda t: abs(int(t[1]) - int(n_nodes)))
        top_k = cand_sorted[: min(32, len(cand_sorted))]
        if len(top_k) == 1:
            gf2 = list(top_k[0][0])[: args.global_dim]
        else:
            gf_mat = np.array([t[0][: args.global_dim] for t in top_k], dtype=np.float32)
            gf2 = np.median(gf_mat, axis=0).astype(float).tolist()

        mm = dict(m)
        mm["node_features"] = nf2.tolist()
        mm["edge_attr"] = ea2.tolist()
        mm["edge_index"] = [list(src), list(dst)]
        mm["global_features"] = gf2
        mm["angle_attr"] = m.get("angle_attr", [])
        mm["angle_triplets"] = m.get("angle_triplets", [])
        mm["num_nodes"] = int(n_nodes)
        mm["num_edges"] = int(len(src))
        mm["num_angles"] = int(len(mm["angle_attr"]))
        out.append(mm)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", args.out_json)
    print(f"edge_norm_stats: r_mean={r_mean:.4f}, r_std={r_std:.4f}, en_mean={en_mean:.4f}, en_std={en_std:.4f}")


if __name__ == "__main__":
    main()
