"""
Ensemble inference for uncertainty estimates on generated materials.

Outputs per-sample mean/std for scalar and Voigt-21 predictions across checkpoints.
"""

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from aim_models.e3_multi_modal import AimMultiModalModel
from datasets.materials_pyg import MaterialsGraphDataset, batch_to_model_io


def load_model(ckpt_path: str, ds: MaterialsGraphDataset, device: str) -> AimMultiModalModel:
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        model = AimMultiModalModel(**ckpt_obj["model_kwargs"]).to(device)
        model.load_state_dict(ckpt_obj["model_state"], strict=True)
    else:
        model = AimMultiModalModel(
            node_dim=ds.in_node,
            edge_dim=ds.in_edge,
            hidden=128,
            n_layers=4,
            cond_dim=ds.in_global,
            out_scalars=8,
            out_voigt=21,
            out_classes=0,
            latent=64,
            n_species=92,
            max_atoms=64,
            use_egnn=True,
        ).to(device)
        model.load_state_dict(ckpt_obj, strict=False)
    model.eval()
    return model


def infer_one(
    model: AimMultiModalModel,
    loader: DataLoader,
    device: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    all_ids: List[str] = []
    all_s: List[List[float]] = []
    all_c: List[List[float]] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = batch_to_model_io(batch, device)
            out = model(**inputs)
            s = torch.nan_to_num(out["pred"]["scalars"]).cpu().numpy()
            c = torch.nan_to_num(out["pred"]["voigt"]).cpu().numpy()
            ids = getattr(batch, "material_id", [None] * len(s))

            for bi in range(len(s)):
                mid = ids[bi] if isinstance(ids, list) else None
                all_ids.append(str(mid))
                all_s.append(s[bi].tolist())
                all_c.append(c[bi].tolist())
    return all_ids, np.array(all_s, dtype=float), np.array(all_c, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="List of checkpoints for ensemble inference.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="predictions/predictions_ensemble.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MaterialsGraphDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_scalars = []
    all_voigt = []
    ref_ids: List[str] = []

    for i, ckpt in enumerate(args.ckpts):
        model = load_model(ckpt, ds, device)
        ids, s, c = infer_one(model, loader, device)
        if i == 0:
            ref_ids = ids
        else:
            if ids != ref_ids:
                raise RuntimeError("Material ordering mismatch across checkpoints.")
        all_scalars.append(s)
        all_voigt.append(c)
        print(f"[{i + 1}/{len(args.ckpts)}] inferred: {ckpt}")

    s_arr = np.stack(all_scalars, axis=0)  # [M, N, 8]
    c_arr = np.stack(all_voigt, axis=0)    # [M, N, 21]

    s_mean = s_arr.mean(axis=0)
    s_std = s_arr.std(axis=0)
    c_mean = c_arr.mean(axis=0)
    c_std = c_arr.std(axis=0)

    rows = []
    for i, mid in enumerate(ref_ids):
        row = {"material_id": mid}
        for k in range(s_mean.shape[1]):
            row[f"s{k}_mean"] = float(s_mean[i, k])
            row[f"s{k}_std"] = float(s_std[i, k])
        for k in range(c_mean.shape[1]):
            row[f"c{k}_mean"] = float(c_mean[i, k])
            row[f"c{k}_std"] = float(c_std[i, k])
        rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out_csv}")
    print(f"n_models={len(args.ckpts)}, n_samples={len(rows)}")


if __name__ == "__main__":
    main()

