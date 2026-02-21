import argparse, os, math, json, random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.materials_pyg import MaterialsGraphDataset, batch_to_model_io
from aim_models.e3_multi_modal import AimMultiModalModel
from utils.metrics import mae_per_column

import matplotlib.pyplot as plt
from collections import defaultdict

def apply_loss_profile(args):
    """
    Optional preset profiles for tensor-focused training.
    "manual" keeps explicit CLI weights unchanged.
    """
    presets = {
        "balanced": {
            "w_voigt": 0.35,
            "w_voigt_eq": 1.2,
            "w_voigt_pd": 0.7,
            "w_voigt_sym": 0.5,
            "w_consistency": 0.25,
            "w_scalar_pos": 0.05,
            "w_tensor_scalar_cons": 0.1,
            "w_class": 0.1,
            "metric_sym_weight": 1.0,
            "metric_eq_weight": 0.7,
            "metric_pd_weight": 0.3,
        },
        "strict_tensor": {
            "w_voigt": 0.45,
            "w_voigt_eq": 2.0,
            "w_voigt_pd": 1.2,
            "w_voigt_sym": 1.0,
            "w_consistency": 0.35,
            "w_scalar_pos": 0.1,
            "w_tensor_scalar_cons": 0.25,
            "w_class": 0.2,
            "metric_sym_weight": 1.5,
            "metric_eq_weight": 1.0,
            "metric_pd_weight": 0.6,
        },
        "stable_generation": {
            "w_voigt": 0.5,
            "w_voigt_eq": 2.2,
            "w_voigt_pd": 1.4,
            "w_voigt_sym": 1.2,
            "w_consistency": 0.4,
            "w_minDist": 0.35,
            "w_nat": 0.35,
            "w_comp_div": 6000.0,
            "w_noble": 7000.0,
            "w_radio": 7000.0,
            "w_scalar_pos": 0.2,
            "w_tensor_scalar_cons": 0.5,
            "w_class": 0.35,
            "metric_sym_weight": 1.5,
            "metric_eq_weight": 1.0,
            "metric_pd_weight": 0.8,
        },
    }
    if args.loss_profile == "manual":
        return
    cfg = presets.get(args.loss_profile)
    if cfg is None:
        return
    for k, v in cfg.items():
        setattr(args, k, v)

def _scalar_from_tensor_like(v, default=-1):
    if v is None:
        return int(default)
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return int(default)
        return int(v.view(-1)[0].item())
    try:
        return int(v)
    except Exception:
        return int(default)

def _group_key_for_leakage_safe_split(d):
    """
    Build a robust proxy key for group-wise split.
    Dataset currently lacks explicit composition/formula fields, so we group by:
      - nelements (or target_num_elements),
      - crystal_system,
      - num_nodes,
      - num_edges.
    """
    ne = _scalar_from_tensor_like(getattr(d, "num_elements", None), default=-1)
    nn = _scalar_from_tensor_like(getattr(d, "num_nodes_graph", None), default=getattr(d, "num_nodes", -1))
    ee = _scalar_from_tensor_like(getattr(d, "num_edges_graph", None), default=getattr(d, "num_edges", -1))
    cs = str(getattr(d, "crystal_system", "") or "").strip().lower()
    return (ne, cs, nn, ee)

def make_group_split_indices(ds, val_ratio=0.10, seed=42):
    groups = defaultdict(list)
    for idx, d in enumerate(ds.data_list):
        groups[_group_key_for_leakage_safe_split(d)].append(idx)

    keys = list(groups.keys())
    rng = random.Random(int(seed))
    rng.shuffle(keys)

    n_total = len(ds)
    n_val_target = max(1, int(round(float(val_ratio) * n_total)))
    val_idx = []
    for k in keys:
        cand = groups[k]
        # Keep at least one sample in train split.
        if len(val_idx) + len(cand) > n_total - 1:
            continue
        val_idx.extend(cand)
        if len(val_idx) >= n_val_target:
            break

    val_set = set(val_idx)
    train_idx = [i for i in range(n_total) if i not in val_set]

    # Safety fallback.
    if len(train_idx) == 0 or len(val_idx) == 0:
        n_val = max(1, int(0.1 * n_total))
        val_idx = list(range(n_val))
        train_idx = list(range(n_val, n_total))

    stats = {
        "n_groups": len(groups),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    return train_idx, val_idx, stats

def checkpoint_score_from_components(args, val_loss, comp_val):
    if args.checkpoint_metric == "val_total":
        return float(val_loss)
    voigt = float(comp_val.get("L_voigt", 0.0))
    if args.checkpoint_metric == "val_voigt":
        return voigt
    if args.checkpoint_metric == "val_stability":
        return (
            voigt
            + float(args.metric_sym_weight) * float(comp_val.get("L_voigt_sym", 0.0))
            + float(args.metric_eq_weight) * float(comp_val.get("L_voigt_eq", 0.0))
            + float(args.metric_pd_weight) * float(comp_val.get("L_voigt_pd", 0.0))
            + float(comp_val.get("L_tensor_scalar_cons", 0.0))
            + float(comp_val.get("L_scalar_pos", 0.0))
            + float(comp_val.get("L_class", 0.0))
        )
    # Balanced Voigt-centric score to avoid degrading symmetry while improving raw fit.
    return (
        voigt
        + float(args.metric_sym_weight) * float(comp_val.get("L_voigt_sym", 0.0))
        + float(args.metric_eq_weight) * float(comp_val.get("L_voigt_eq", 0.0))
        + float(args.metric_pd_weight) * float(comp_val.get("L_voigt_pd", 0.0))
    )

def load_norm_stats(npz_path):
    if not npz_path or not os.path.exists(npz_path):
        return None, None, None, None
    try:
        stats = np.load(npz_path)
        scalar_mean = stats["scalar_mean"].tolist() if "scalar_mean" in stats else None
        scalar_std = stats["scalar_std"].tolist() if "scalar_std" in stats else None
        voigt_mean = stats["voigt_mean"].tolist() if "voigt_mean" in stats else None
        voigt_std = stats["voigt_std"].tolist() if "voigt_std" in stats else None
        return scalar_mean, scalar_std, voigt_mean, voigt_std
    except Exception as e:
        print(f"[warn] failed to load norm stats from {npz_path}: {e}")
        return None, None, None, None

def compute_target_stats_from_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return None, None, None, None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        print(f"[warn] failed to read raw stats json {json_path}: {e}")
        return None, None, None, None

    s_vals = []
    c_vals = []
    for r in rows:
        s = r.get("targets_scalars")
        c = r.get("targets_voigt21")
        if isinstance(s, list) and len(s) == 8:
            try:
                s_vals.append([float(x) for x in s])
            except Exception:
                pass
        if isinstance(c, list) and len(c) == 21:
            try:
                c_vals.append([float(x) for x in c])
            except Exception:
                pass
    if not s_vals or not c_vals:
        return None, None, None, None

    s_arr = np.array(s_vals, dtype=float)
    c_arr = np.array(c_vals, dtype=float)
    s_mean = np.nanmean(s_arr, axis=0).tolist()
    s_std = np.nanstd(s_arr, axis=0).tolist()
    c_mean = np.nanmean(c_arr, axis=0).tolist()
    c_std = np.nanstd(c_arr, axis=0).tolist()
    return s_mean, s_std, c_mean, c_std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to JSON dataset")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--latent", type=int, default=64)                 # <— new: latent size knob
    ap.add_argument("--logdir", type=str, default="runs/aim")
    ap.add_argument("--checkpoint", type=str, default="aim_best.pt")
    ap.add_argument(
        "--checkpoint_metric",
        type=str,
        default="val_total",
        choices=["val_total", "val_voigt", "val_voigt_balanced", "val_stability"],
        help="Metric used to choose best checkpoint.",
    )
    ap.add_argument("--metric_sym_weight", type=float, default=1.0)
    ap.add_argument("--metric_eq_weight", type=float, default=0.5)
    ap.add_argument("--metric_pd_weight", type=float, default=0.0)
    # Optional: override prior weights from CLI (else defaults below)
    ap.add_argument("--w_kld", type=float, default=0.05)
    ap.add_argument("--w_minDist", type=float, default=0.2)
    ap.add_argument("--w_recip", type=float, default=0.05)
    ap.add_argument("--w_pred", type=float, default=1.0)
    ap.add_argument("--w_voigt", type=float, default=0.3)
    ap.add_argument("--w_voigt_eq", type=float, default=1.0)
    ap.add_argument("--w_voigt_pd", type=float, default=0.5)
    ap.add_argument("--w_voigt_sym", type=float, default=0.3)
    ap.add_argument("--w_consistency", type=float, default=0.2)
    ap.add_argument("--w_nat", type=float, default=0.2)
    ap.add_argument("--w_scalar_pos", type=float, default=0.05)
    ap.add_argument("--w_tensor_scalar_cons", type=float, default=0.1)
    ap.add_argument("--w_class", type=float, default=0.1)
    ap.add_argument("--w_comp_div", type=float, default=5000.0)
    ap.add_argument("--w_noble", type=float, default=5000.0)
    ap.add_argument("--w_radio", type=float, default=5000.0)
    ap.add_argument(
        "--loss_profile",
        type=str,
        default="balanced",
        choices=["manual", "balanced", "strict_tensor", "stable_generation"],
        help="Preset for tensor-focused weights (manual keeps explicit weights).",
    )
    ap.add_argument(
        "--hard_symmetry_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply crystal-system hard masks to Voigt outputs.",
    )
    ap.add_argument("--max_elements_loss", type=int, default=4,
                    help="Target max unique elements for composition diversity penalty.")
    ap.add_argument(
        "--use_stability_class_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auxiliary binary stability classification head when labels are available.",
    )
    ap.add_argument("--norm_stats_npz", type=str, default="normalization_stats.npz")
    ap.add_argument(
        "--raw_stats_data",
        type=str,
        default="processed_data_filtered/all_materials_data.json",
        help="Fallback raw (unnormalized) JSON for scalar/voigt mean/std if NPZ stats are invalid.",
    )
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="", help="resume from a checkpoint path (optional)")

    args = ap.parse_args()
    apply_loss_profile(args)
    print(
        f"[loss_profile] {args.loss_profile} | "
        f"voigt={args.w_voigt} eq={args.w_voigt_eq} pd={args.w_voigt_pd} sym={args.w_voigt_sym} "
        f"cons={args.w_consistency} scalar_pos={args.w_scalar_pos} "
        f"tensor_scalar_cons={args.w_tensor_scalar_cons} class={args.w_class} "
        f"hard_symmetry_mask={bool(args.hard_symmetry_mask)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MaterialsGraphDataset(args.data)
    n_node, n_edge, n_global = ds.in_node, ds.in_edge, ds.in_global
    scalar_mean, scalar_std, voigt_mean, voigt_std = load_norm_stats(args.norm_stats_npz)
    scalar_ok = isinstance(scalar_mean, list) and isinstance(scalar_std, list) and len(scalar_mean) == 8 and len(scalar_std) == 8
    voigt_ok = isinstance(voigt_mean, list) and isinstance(voigt_std, list) and len(voigt_mean) == 21 and len(voigt_std) == 21
    if not (scalar_ok and voigt_ok):
        print(f"[warn] normalization stats missing/incomplete at {args.norm_stats_npz}; trying fallback from {args.raw_stats_data}")
        fs_m, fs_s, fv_m, fv_s = compute_target_stats_from_json(args.raw_stats_data)
        if fs_m is not None and fv_m is not None:
            scalar_mean, scalar_std, voigt_mean, voigt_std = fs_m, fs_s, fv_m, fv_s
            print("[info] loaded fallback physical stats from raw JSON.")
        else:
            print("[warn] fallback raw stats unavailable; physical consistency losses may be inactive.")

    use_egnn = ds.pos_fraction > 0.5  # enable EGNN if majority have coordinates
    if not use_egnn:
        print(f"[info] No positions in dataset (pos_fraction={ds.pos_fraction:.2f}). "
              f"Falling back to invariant backbone.")

    # Leakage-safe split by grouped proxy key (composition/system/graph-size signature).
    train_idx, val_idx, split_stats = make_group_split_indices(
        ds, val_ratio=args.val_ratio, seed=args.split_seed
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    print(
        f"[split] train={split_stats['n_train']} val={split_stats['n_val']} "
        f"groups={split_stats['n_groups']} val_ratio={args.val_ratio:.3f} seed={args.split_seed}"
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = AimMultiModalModel(
        node_dim=n_node, edge_dim=n_edge, hidden=args.hidden, n_layers=args.layers,
        cond_dim=n_global, out_scalars=8, out_voigt=21, out_classes=(2 if args.use_stability_class_head else 0),
        latent=args.latent, n_species=92, max_atoms=64,
        use_egnn=use_egnn, use_motif_pool=True,
        enforce_spd_voigt=True, spd_eps=1e-4, hard_symmetry_mask=bool(args.hard_symmetry_mask),
        max_elements_loss=args.max_elements_loss,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        voigt_mean=voigt_mean,
        voigt_std=voigt_std,
        # If your model supports internal prior weights, you can also pass:
        # unsup_priors=True,
        # prior_weights={"kld": args.w_kld, "minDist": args.w_minDist, "recip": args.w_recip},
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 1
    best_score = math.inf
    best_val = math.inf

    if args.resume:
        ckpt_obj = torch.load(args.resume, map_location=device)
        if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
            try:
                model.load_state_dict(ckpt_obj["model_state"], strict=True)
            except RuntimeError as e:
                print(f"[warn] strict resume load failed, falling back to strict=False: {e}")
                model.load_state_dict(ckpt_obj["model_state"], strict=False)
            if "opt_state" in ckpt_obj:
                opt.load_state_dict(ckpt_obj["opt_state"])
            start_epoch = int(ckpt_obj.get("epoch", 0)) + 1
            best_val = float(ckpt_obj.get("best_val", best_val))
            if "best_score" in ckpt_obj:
                best_score = float(ckpt_obj.get("best_score", best_score))
            elif args.checkpoint_metric == "val_total":
                best_score = best_val
            print(
                f"[resume] loaded {args.resume} (start_epoch={start_epoch}, best_val={best_val}, "
                f"best_score[{args.checkpoint_metric}]={best_score})"
            )
        else:
            # backward compatible: raw state_dict
            model.load_state_dict(ckpt_obj, strict=True)
            print(f"[resume] loaded raw state_dict from {args.resume}")


    writer = SummaryWriter(args.logdir)

    # ---- Step 1: loss weights + logs store ----
    W = dict(
    pred=args.w_pred,
    voigt=args.w_voigt,
    voigt_eq=args.w_voigt_eq,
    voigt_pd=args.w_voigt_pd,
    voigt_sym=args.w_voigt_sym,
    consistency=args.w_consistency,
    scalar_pos=args.w_scalar_pos,
    tensor_scalar_cons=args.w_tensor_scalar_cons,
    cls=args.w_class,
    kld=args.w_kld,
    minDist=args.w_minDist,
    recip=args.w_recip,
    nat=args.w_nat,
    composition_diversity=args.w_comp_div,
    noble_gas=args.w_noble,
    radioactive=args.w_radio,
)

    # per-epoch logs (for plotting/CSV)
    logs = defaultdict(list)

    for epoch in range(start_epoch, args.epochs+1):
        # ================ TRAIN ================
        model.train()
        total = 0.0
        with torch.no_grad():
            p0 = next(model.parameters()).view(-1)[0].item()
        print(f"[epoch {epoch}] param0 = {p0:.6e}")

        # component accumulators (unweighted) for epoch-averages
        comp_train = defaultdict(float)
        n_graphs_train = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"train {epoch:03d}")):
            inputs, targets = batch_to_model_io(batch, device)
            
            # Add batch_nat for L_nat loss
            batch_nat = torch.tensor([g.num_nodes for g in batch.to_data_list()], 
                                    dtype=torch.float32, device=device)
            
            out = model(**inputs, **targets, batch_nat=batch_nat)
            losses = out["losses"]

            # KL anneal (linearly to epoch 30)
            kl_scale = min(1.0, epoch/30.0) * W["kld"]

            l = 0.0
            # Use .get(..., 0) but keep tensors on device
            L_pred     = losses.get("L_pred_scalar", torch.tensor(0., device=device))
            L_voigt    = losses.get("L_voigt",        torch.tensor(0., device=device))
            L_voigt_eq = losses.get("L_voigt_eq",     torch.tensor(0., device=device))
            L_voigt_pd = losses.get("L_voigt_pd",     torch.tensor(0., device=device))
            L_voigt_sym= losses.get("L_voigt_sym",    torch.tensor(0., device=device))
            L_scalar_pos = losses.get("L_scalar_pos", torch.tensor(0., device=device))
            L_tensor_scalar_cons = losses.get("L_tensor_scalar_cons", torch.tensor(0., device=device))
            L_class    = losses.get("L_class",        torch.tensor(0., device=device))
            L_cons     = losses.get("L_consistency",  torch.tensor(0., device=device))
            L_kld      = losses.get("L_kld",          torch.tensor(0., device=device))
            L_minDist  = losses.get("L_minDist",      torch.tensor(0., device=device))
            L_recip    = losses.get("L_recip",        torch.tensor(0., device=device))
            L_nat      = losses.get("L_nat",          torch.tensor(0., device=device))
            L_comp_div = losses.get("L_composition_diversity", torch.tensor(0., device=device))
            L_noble    = losses.get("L_noble_gas",    torch.tensor(0., device=device))
            L_radio    = losses.get("L_radioactive",  torch.tensor(0., device=device))

            l += W["pred"]     * L_pred
            l += W["voigt"]    * L_voigt
            l += W["voigt_eq"] * L_voigt_eq
            l += W["voigt_pd"] * L_voigt_pd
            l += W["voigt_sym"] * L_voigt_sym
            l += W["scalar_pos"] * L_scalar_pos
            l += W["tensor_scalar_cons"] * L_tensor_scalar_cons
            l += W["cls"]      * L_class
            l += W["consistency"] * L_cons
            l += kl_scale      * L_kld
            l += W["minDist"]  * L_minDist
            l += W["recip"]    * L_recip
            l += W.get("nat", 0.2) * L_nat  # Use 0.2 as default weight
            l += W["composition_diversity"] * L_comp_div 
            l += W["noble_gas"]             * L_noble    
            l += W["radioactive"]           * L_radio 

            opt.zero_grad(set_to_none=True)
            l.backward()

            # ---- NaN/Inf gradient guard ----
            bad_grad = False
            for p in model.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break

            if (not torch.isfinite(l)) or bad_grad:
                print(f"[skip] non-finite loss/grad at epoch={epoch}, batch={batch_idx}")
                opt.zero_grad(set_to_none=True)
                continue
            # --------------------------------

            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()


            # book-keeping
            bsz_graphs = batch.num_graphs
            total += float(l.item()) * bsz_graphs
            n_graphs_train += bsz_graphs
            # accumulate unweighted components for reporting
            comp_train["L_pred_scalar"] += float(L_pred.item()) * bsz_graphs
            comp_train["L_voigt"]       += float(L_voigt.item()) * bsz_graphs
            comp_train["L_voigt_eq"]    += float(L_voigt_eq.item()) * bsz_graphs
            comp_train["L_voigt_pd"]    += float(L_voigt_pd.item()) * bsz_graphs
            comp_train["L_voigt_sym"]   += float(L_voigt_sym.item()) * bsz_graphs
            comp_train["L_scalar_pos"]  += float(L_scalar_pos.item()) * bsz_graphs
            comp_train["L_tensor_scalar_cons"] += float(L_tensor_scalar_cons.item()) * bsz_graphs
            comp_train["L_class"]       += float(L_class.item()) * bsz_graphs
            comp_train["L_consistency"] += float(L_cons.item()) * bsz_graphs
            comp_train["L_kld"]         += float(L_kld.item()) * bsz_graphs
            comp_train["L_minDist"]     += float(L_minDist.item()) * bsz_graphs
            comp_train["L_recip"]       += float(L_recip.item()) * bsz_graphs
            comp_train["L_nat"]         += float(L_nat.item()) * bsz_graphs
            comp_train["L_composition_diversity"] += float(L_comp_div.item()) * bsz_graphs
            comp_train["L_noble_gas"]             += float(L_noble.item()) * bsz_graphs
            comp_train["L_radioactive"]           += float(L_radio.item()) * bsz_graphs

        train_loss = total / max(1, n_graphs_train)
        # normalize component means
        for k in list(comp_train.keys()):
            comp_train[k] = comp_train[k] / max(1, n_graphs_train)

        # ================ VAL ================
        model.eval()
        val_total = 0.0
        comp_val = defaultdict(float)
        n_graphs_val = 0
        scalar_mae = None

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val"):
                inputs, targets = batch_to_model_io(batch, device)
                
                # Add batch_nat for L_nat loss
                batch_nat = torch.tensor([g.num_nodes for g in batch.to_data_list()], 
                                        dtype=torch.float32, device=device)
                
                out = model(**inputs, **targets, batch_nat=batch_nat)
                losses = out["losses"]

                L_pred     = losses.get("L_pred_scalar", torch.tensor(0., device=device))
                L_voigt    = losses.get("L_voigt",        torch.tensor(0., device=device))
                L_voigt_eq = losses.get("L_voigt_eq",     torch.tensor(0., device=device))
                L_voigt_pd = losses.get("L_voigt_pd",     torch.tensor(0., device=device))
                L_voigt_sym= losses.get("L_voigt_sym",    torch.tensor(0., device=device))
                L_scalar_pos = losses.get("L_scalar_pos", torch.tensor(0., device=device))
                L_tensor_scalar_cons = losses.get("L_tensor_scalar_cons", torch.tensor(0., device=device))
                L_class    = losses.get("L_class",        torch.tensor(0., device=device))
                L_cons     = losses.get("L_consistency",  torch.tensor(0., device=device))
                L_kld      = losses.get("L_kld",          torch.tensor(0., device=device))
                L_minDist  = losses.get("L_minDist",      torch.tensor(0., device=device))
                L_recip    = losses.get("L_recip",        torch.tensor(0., device=device))
                L_nat      = losses.get("L_nat",          torch.tensor(0., device=device))
                L_comp_div = losses.get("L_composition_diversity", torch.tensor(0., device=device))
                L_noble    = losses.get("L_noble_gas",    torch.tensor(0., device=device))
                L_radio    = losses.get("L_radioactive",  torch.tensor(0., device=device))

                l = 0.0
                l += W["pred"]     * L_pred
                l += W["voigt"]    * L_voigt
                l += W["voigt_eq"] * L_voigt_eq
                l += W["voigt_pd"] * L_voigt_pd
                l += W["voigt_sym"] * L_voigt_sym
                l += W["scalar_pos"] * L_scalar_pos
                l += W["tensor_scalar_cons"] * L_tensor_scalar_cons
                l += W["cls"]      * L_class
                l += W["consistency"] * L_cons
                l += W["kld"]      * L_kld
                l += W["minDist"]  * L_minDist
                l += W["recip"]    * L_recip
                l += W.get("nat", 0.2) * L_nat  # Use 0.2 as default weight
                l += W["composition_diversity"] * L_comp_div 
                l += W["noble_gas"]             * L_noble    
                l += W["radioactive"]           * L_radio 

                bsz_graphs = batch.num_graphs
                val_total += float(l.item()) * bsz_graphs
                n_graphs_val += bsz_graphs

                comp_val["L_pred_scalar"] += float(L_pred.item()) * bsz_graphs
                comp_val["L_voigt"]       += float(L_voigt.item()) * bsz_graphs
                comp_val["L_voigt_eq"]    += float(L_voigt_eq.item()) * bsz_graphs
                comp_val["L_voigt_pd"]    += float(L_voigt_pd.item()) * bsz_graphs
                comp_val["L_voigt_sym"]   += float(L_voigt_sym.item()) * bsz_graphs
                comp_val["L_scalar_pos"]  += float(L_scalar_pos.item()) * bsz_graphs
                comp_val["L_tensor_scalar_cons"] += float(L_tensor_scalar_cons.item()) * bsz_graphs
                comp_val["L_class"]       += float(L_class.item()) * bsz_graphs
                comp_val["L_consistency"] += float(L_cons.item()) * bsz_graphs
                comp_val["L_kld"]         += float(L_kld.item()) * bsz_graphs
                comp_val["L_minDist"]     += float(L_minDist.item()) * bsz_graphs
                comp_val["L_recip"]       += float(L_recip.item()) * bsz_graphs
                comp_val["L_nat"]         += float(L_nat.item()) * bsz_graphs
                comp_val["L_composition_diversity"] += float(L_comp_div.item()) * bsz_graphs
                comp_val["L_noble_gas"]             += float(L_noble.item()) * bsz_graphs
                comp_val["L_radioactive"]           += float(L_radio.item()) * bsz_graphs

                if "y_scalars" in targets and out["pred"]["scalars"].shape == targets["y_scalars"].shape:
                    scalar_mae = mae_per_column(out["pred"]["scalars"], targets["y_scalars"])

        val_loss = val_total / max(1, n_graphs_val)
        for k in list(comp_val.keys()):
            comp_val[k] = comp_val[k] / max(1, n_graphs_val)
        score = checkpoint_score_from_components(args, val_loss, comp_val)
        best_val = min(best_val, val_loss)

        # ---- TensorBoard scalars ----
        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/val_total",   val_loss,   epoch)
        writer.add_scalar(f"metric/{args.checkpoint_metric}", score, epoch)
        for name in ["L_pred_scalar","L_voigt","L_voigt_eq","L_voigt_pd","L_voigt_sym",
                     "L_scalar_pos","L_tensor_scalar_cons","L_class","L_consistency",
                     "L_kld","L_minDist","L_recip",
                     "L_composition_diversity","L_noble_gas","L_radioactive"]:
            writer.add_scalar(f"train/{name}", comp_train.get(name, 0.0), epoch)
            writer.add_scalar(f"val/{name}",   comp_val.get(name, 0.0),   epoch)
        if scalar_mae is not None:
            for i, v in enumerate(scalar_mae):
                writer.add_scalar(f"mae/scalar_{i}", v, epoch)

        # ---- Console summary (includes components) ----
        print(
            f"[epoch {epoch:03d}] "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"score[{args.checkpoint_metric}]={score:.4f} | "
            f"KLD_diag(T/V)={comp_train['L_kld']:.4f}/{comp_val['L_kld']:.4f} | "
            f"pred(T/V)={comp_train['L_pred_scalar']:.4f}/{comp_val['L_pred_scalar']:.4f} | "
            f"voigt(T/V)={comp_train['L_voigt']:.4f}/{comp_val['L_voigt']:.4f} | "
            f"voigt_pd(T/V)={comp_train.get('L_voigt_pd', 0):.4f}/{comp_val.get('L_voigt_pd', 0):.4f} | "
            f"t2s(T/V)={comp_train.get('L_tensor_scalar_cons', 0):.4f}/{comp_val.get('L_tensor_scalar_cons', 0):.4f} | "
            f"spos(T/V)={comp_train.get('L_scalar_pos', 0):.4f}/{comp_val.get('L_scalar_pos', 0):.4f} | "
            f"cls(T/V)={comp_train.get('L_class', 0):.4f}/{comp_val.get('L_class', 0):.4f} | "
            f"cons(T/V)={comp_train.get('L_consistency', 0):.4f}/{comp_val.get('L_consistency', 0):.4f} | "
            f"nat(T/V)={comp_train.get('L_nat', 0):.4f}/{comp_val.get('L_nat', 0):.4f} | "
            f"comp(T/V)={comp_train.get('L_composition_diversity', 0):.4f}/{comp_val.get('L_composition_diversity', 0):.4f} | "
            f"noble(T/V)={comp_train.get('L_noble_gas', 0):.4f}/{comp_val.get('L_noble_gas', 0):.4f}"
        )


        # ---- Save best ----
        if score < best_score:
            best_score = score
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True) if os.path.dirname(args.checkpoint) else None

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "checkpoint_metric": args.checkpoint_metric,
                    "best_val": best_val,
                    "model_kwargs": {
                        "node_dim": n_node,
                        "edge_dim": n_edge,
                        "hidden": args.hidden,
                        "n_layers": args.layers,
                        "cond_dim": n_global,
                        "out_scalars": 8,
                        "out_voigt": 21,
                        "out_classes": (2 if args.use_stability_class_head else 0),
                        "latent": args.latent,
                        "n_species": 92,
                        "max_atoms": 64,
                        "use_egnn": use_egnn,
                        "use_motif_pool": True,
                        "enforce_spd_voigt": True,
                        "spd_eps": 1e-4,
                        "hard_symmetry_mask": bool(args.hard_symmetry_mask),
                        "max_elements_loss": args.max_elements_loss,
                        "scalar_mean": scalar_mean,
                        "scalar_std": scalar_std,
                        "voigt_mean": voigt_mean,
                        "voigt_std": voigt_std,
                    },
                },
                args.checkpoint
            )
            print(f"  -> saved {args.checkpoint}")

        # ---- Step 2: accumulate epoch logs for plotting/CSV ----
        logs["epoch"].append(epoch)
        logs["train_total"].append(train_loss)
        logs["val_total"].append(val_loss)
        for name in ["L_pred_scalar","L_voigt","L_voigt_eq","L_voigt_pd","L_voigt_sym",
                     "L_scalar_pos","L_tensor_scalar_cons","L_class","L_consistency",
                     "L_kld","L_minDist","L_recip"]:
            logs[f"train_{name}"].append(comp_train.get(name, 0.0))
            logs[f"val_{name}"].append(comp_val.get(name, 0.0))

    print(f"Done. Best val={best_val:.6f} | best_score[{args.checkpoint_metric}]={best_score:.6f}")

    # ---- Step 2: Plot after training ----
    try:
        plt.figure(figsize=(9,6))
        plt.plot(logs["epoch"], logs["train_total"], label="train total")
        plt.plot(logs["epoch"], logs["val_total"],   label="val total", linestyle="--")
        for name in ["L_pred_scalar","L_voigt","L_kld","L_minDist"]:
            if f"train_{name}" in logs:
                plt.plot(logs["epoch"], logs[f"train_{name}"], label=f"train {name}")
            if f"val_{name}" in logs:
                plt.plot(logs["epoch"], logs[f"val_{name}"], linestyle="--", label=f"val {name}")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig("loss_components.png", dpi=200)
        # plt.show()  # uncomment if you want to display a window
        print("Saved plot: loss_components.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # ---- Step 3: CSV export ----
    try:
        import pandas as pd
        pd.DataFrame(logs).to_csv("loss_log.csv", index=False)
        print("Saved CSV: loss_log.csv")
    except Exception:
        # Fallback: write minimal CSV manually
        try:
            keys = list(logs.keys())
            with open("loss_log.csv", "w") as f:
                f.write(",".join(keys) + "\n")
                for i in range(len(logs["epoch"])):
                    row = []
                    for k in keys:
                        vlist = logs[k]
                        row.append(str(vlist[i]) if i < len(vlist) else "")
                    f.write(",".join(row) + "\n")
            print("Saved CSV (manual): loss_log.csv")
        except Exception as e:
            print(f"[warn] CSV export failed: {e}")

if __name__ == "__main__":
    main()
