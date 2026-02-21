"""
Run a strict end-to-end generation + property inference + deep validation pipeline.

This script automates:
1) conditional generation (2/3/4-element),
2) featurization aligned with training distribution,
3) property inference and merge,
4) deep structural/mechanical validation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: str) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run] {pretty}")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_sets(text: str) -> List[int]:
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v < 1:
            raise ValueError(f"Invalid set value: {v}")
        out.append(v)
    if not out:
        raise ValueError("No valid sets were provided.")
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Model checkpoint for generation and inference.")
    ap.add_argument("--python", default=sys.executable, help="Python executable.")
    ap.add_argument("--sets", default="2,3,4", help="Comma-separated target element counts.")
    ap.add_argument("--n_per_set", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="densityaware_strict")

    ap.add_argument("--natoms_mode", default="dataset_sample", choices=["fixed", "decoded", "dataset_sample"])
    ap.add_argument("--natoms_dataset", default="processed_data_filtered/all_materials_data.json")
    ap.add_argument("--natoms_min", type=int, default=8)
    ap.add_argument("--natoms_max", type=int, default=64)
    ap.add_argument("--fixed_atoms", type=int, default=64)

    ap.add_argument("--min_distance", type=float, default=1.2)
    ap.add_argument("--min_volume", type=float, default=100.0)
    ap.add_argument("--max_volume", type=float, default=5000.0)
    ap.add_argument("--target_density_min", type=float, default=1.0)
    ap.add_argument("--target_density_max", type=float, default=15.0)
    ap.add_argument("--vpa_min", type=float, default=2.0)
    ap.add_argument("--vpa_max", type=float, default=80.0)
    ap.add_argument("--density_quantile_low", type=float, default=0.05)
    ap.add_argument("--density_quantile_high", type=float, default=0.95)
    ap.add_argument("--density_source_json", default="datasets/mp_summary_filtered.json")

    ap.add_argument("--cn_cutoff", type=float, default=3.0)
    ap.add_argument("--cn_mean_min", type=float, default=0.5)
    ap.add_argument("--cn_mean_max", type=float, default=20.0)
    ap.add_argument("--cn_max", type=int, default=32)
    ap.add_argument("--repair_retries", type=int, default=8)
    ap.add_argument("--consistency_relerr_max", type=float, default=0.5)

    ap.add_argument("--norm_stats_npz", default="normalization_stats.npz")
    ap.add_argument("--train_summary_json", default="datasets/mp_summary_filtered.json")
    ap.add_argument("--reference_data", default="processed_data_filtered/all_materials_data.json")
    ap.add_argument("--elements_csv", default="datasets/PubChemElements_all.csv")
    ap.add_argument("--topk", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.getcwd()
    sets = parse_sets(args.sets)
    summaries: Dict[str, dict] = {}

    for ne in sets:
        key = f"{ne}el"
        base = f"{key}_{args.tag}"
        cif_dir = f"generated_cifs_{base}"
        gen_json = f"generated_materials_{base}.json"
        feat_json = f"generated_materials_{base}_featurized_real.json"
        pred_csv = f"predictions/{base}.csv"
        merged_json = f"generated_materials_{base}_with_predictions_real.json"
        val_json = f"validation_{base}_deep.json"
        val_summary = f"validation_{base}_deep_summary.json"
        val_topk = f"candidates_{base}_strict_novel_unique.csv"

        run_cmd(
            [
                args.python,
                "generate_structures.py",
                "--ckpt",
                args.ckpt,
                "--n",
                str(args.n_per_set),
                "--batch_size",
                str(args.batch_size),
                "--temperature",
                str(args.temperature),
                "--seed",
                str(args.seed + ne),
                "--natoms_mode",
                args.natoms_mode,
                "--natoms_dataset",
                args.natoms_dataset,
                "--natoms_min",
                str(args.natoms_min),
                "--natoms_max",
                str(args.natoms_max),
                "--fixed_atoms",
                str(args.fixed_atoms),
                "--target_num_elements",
                str(ne),
                "--enforce_exact_num_elements",
                "--min_distance",
                str(args.min_distance),
                "--min_volume",
                str(args.min_volume),
                "--max_volume",
                str(args.max_volume),
                "--density_source_json",
                args.density_source_json,
                "--target_density_min",
                str(args.target_density_min),
                "--target_density_max",
                str(args.target_density_max),
                "--density_quantile_low",
                str(args.density_quantile_low),
                "--density_quantile_high",
                str(args.density_quantile_high),
                "--forbid_noble_gas",
                "--forbid_radioactive",
                "--require_neutrality_guess",
                "--cn_cutoff",
                str(args.cn_cutoff),
                "--vpa_min",
                str(args.vpa_min),
                "--vpa_max",
                str(args.vpa_max),
                "--cn_mean_min",
                str(args.cn_mean_min),
                "--cn_mean_max",
                str(args.cn_mean_max),
                "--cn_max",
                str(args.cn_max),
                "--repair_retries",
                str(args.repair_retries),
                "--outdir",
                cif_dir,
                "--out_json",
                gen_json,
            ],
            cwd=root,
        )

        run_cmd(
            [
                args.python,
                "make_featurized_stub.py",
                "--in_json",
                gen_json,
                "--out_json",
                feat_json,
                "--reference_data",
                args.reference_data,
                "--elements_csv",
                args.elements_csv,
            ],
            cwd=root,
        )

        run_cmd(
            [
                args.python,
                "infer_properties.py",
                "--data",
                feat_json,
                "--ckpt",
                args.ckpt,
                "--batch_size",
                str(args.batch_size),
                "--out_csv",
                pred_csv,
            ],
            cwd=root,
        )

        run_cmd(
            [
                args.python,
                "merge_predictions.py",
                "--in_json",
                feat_json,
                "--pred_csv",
                pred_csv,
                "--out_json",
                merged_json,
            ],
            cwd=root,
        )

        run_cmd(
            [
                args.python,
                "validate_generated_depth.py",
                "--in_json",
                merged_json,
                "--cif_dir",
                cif_dir,
                "--out_json",
                val_json,
                "--out_summary_json",
                val_summary,
                "--out_topk_csv",
                val_topk,
                "--norm_stats_npz",
                args.norm_stats_npz,
                "--train_summary_json",
                args.train_summary_json,
                "--min_distance",
                str(args.min_distance),
                "--density_min",
                str(args.target_density_min),
                "--density_max",
                str(args.target_density_max),
                "--vpa_min",
                str(args.vpa_min),
                "--vpa_max",
                str(args.vpa_max),
                "--cn_cutoff",
                str(args.cn_cutoff),
                "--cn_mean_min",
                str(args.cn_mean_min),
                "--cn_mean_max",
                str(args.cn_mean_max),
                "--cn_max",
                str(args.cn_max),
                "--require_neutrality_guess",
                "--consistency_relerr_max",
                str(args.consistency_relerr_max),
                "--topk",
                str(args.topk),
            ],
            cwd=root,
        )

        with open(val_summary, "r", encoding="utf-8") as f:
            summaries[key] = json.load(f)

    aggregate_path = f"validation_{args.tag}_aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"sets={','.join(str(x) for x in sets)}")
    print(f"aggregate_summary={aggregate_path}")
    for key in summaries:
        sp = summaries[key].get("strict_pass", {})
        pd = summaries[key].get("is_pd", {})
        cn = summaries[key].get("pass_cn_reasonable", {})
        neu = summaries[key].get("pass_neutrality_guess", {})
        print(
            f"{key}: strict={sp.get('count', 0)}/{sp.get('total', 0)} "
            f"({100.0 * float(sp.get('rate') or 0.0):.2f}%), "
            f"pd={100.0 * float(pd.get('rate') or 0.0):.2f}%, "
            f"cn={100.0 * float(cn.get('rate') or 0.0):.2f}%, "
            f"neutral={100.0 * float(neu.get('rate') or 0.0):.2f}%"
        )


if __name__ == "__main__":
    main()
