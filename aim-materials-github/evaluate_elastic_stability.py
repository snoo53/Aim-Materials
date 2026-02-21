import argparse
import json
from collections import Counter, defaultdict

import numpy as np


VOIGT21_IDXS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5),
    (5, 5),
]


def voigt21_to_c6(voigt21):
    if not isinstance(voigt21, list) or len(voigt21) != 21:
        return None
    c = np.zeros((6, 6), dtype=float)
    for k, (i, j) in enumerate(VOIGT21_IDXS):
        v = float(voigt21[k])
        c[i, j] = v
        c[j, i] = v
    return c


def maybe_denormalize_voigt21(voigt21, voigt_mean=None, voigt_std=None):
    """
    If normalization stats are provided, map normalized Voigt-21 back to physical units.
    Supports scalar mean/std in npz (current normalize_data.py behavior).
    """
    if voigt_mean is None or voigt_std is None:
        return voigt21
    out = []
    for x in voigt21:
        out.append(float(x) * float(voigt_std) + float(voigt_mean))
    return out


def born_checks(c, crystal_system):
    cs = (crystal_system or "").strip().lower()
    c11, c22, c33 = c[0, 0], c[1, 1], c[2, 2]
    c12, c13, c23 = c[0, 1], c[0, 2], c[1, 2]
    c44, c55, c66 = c[3, 3], c[4, 4], c[5, 5]

    checks = []

    if cs == "cubic":
        checks = [
            ("C11-C12>0", c11 - c12 > 0.0),
            ("C11+2C12>0", c11 + 2.0 * c12 > 0.0),
            ("C44>0", c44 > 0.0),
        ]
    elif cs in ("hexagonal", "trigonal"):
        checks = [
            ("C11-|C12|>0", c11 - abs(c12) > 0.0),
            ("2C13^2<C33(C11+C12)", 2.0 * c13 * c13 < c33 * (c11 + c12)),
            ("C44>0", c44 > 0.0),
        ]
    elif cs == "tetragonal":
        checks = [
            ("C11-|C12|>0", c11 - abs(c12) > 0.0),
            ("2C13^2<C33(C11+C12)", 2.0 * c13 * c13 < c33 * (c11 + c12)),
            ("C44>0", c44 > 0.0),
            ("C66>0", c66 > 0.0),
        ]
    elif cs == "orthorhombic":
        checks = [
            ("C11>0", c11 > 0.0),
            ("C22>0", c22 > 0.0),
            ("C33>0", c33 > 0.0),
            ("C44>0", c44 > 0.0),
            ("C55>0", c55 > 0.0),
            ("C66>0", c66 > 0.0),
            ("C11+C22-2C12>0", c11 + c22 - 2.0 * c12 > 0.0),
            ("C11+C33-2C13>0", c11 + c33 - 2.0 * c13 > 0.0),
            ("C22+C33-2C23>0", c22 + c33 - 2.0 * c23 > 0.0),
            ("C11+C22+C33+2(C12+C13+C23)>0", c11 + c22 + c33 + 2.0 * (c12 + c13 + c23) > 0.0),
        ]
    else:
        return None, []

    failed = [name for name, ok in checks if not ok]
    return len(failed) == 0, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Dataset with targets_voigt21")
    ap.add_argument("--out_json", default="", help="Optional per-sample stability report")
    ap.add_argument("--eig_tol", type=float, default=1e-6, help="Min eigenvalue tolerance for PD")
    ap.add_argument(
        "--denorm_stats_npz",
        default="",
        help="Optional npz with voigt_mean/voigt_std to de-normalize targets_voigt21 before checks.",
    )
    args = ap.parse_args()

    with open(args.in_json, "r") as f:
        mats = json.load(f)

    voigt_mean = None
    voigt_std = None
    if args.denorm_stats_npz:
        stats = np.load(args.denorm_stats_npz)
        if "voigt_mean" not in stats or "voigt_std" not in stats:
            raise ValueError(f"{args.denorm_stats_npz} missing voigt_mean/voigt_std")
        voigt_mean = float(np.array(stats["voigt_mean"]).reshape(-1)[0])
        voigt_std = float(np.array(stats["voigt_std"]).reshape(-1)[0])

    rows = []
    stats = Counter()
    by_cs = defaultdict(Counter)

    for m in mats:
        mid = m.get("material_id", "")
        cs = m.get("crystal_system", "")
        v = m.get("targets_voigt21")
        v = maybe_denormalize_voigt21(v, voigt_mean, voigt_std) if v is not None else v
        c = voigt21_to_c6(v)

        if c is None:
            stats["missing_voigt"] += 1
            continue

        stats["with_voigt"] += 1

        csym = 0.5 * (c + c.T)
        eigvals = np.linalg.eigvalsh(csym)
        min_eig = float(np.min(eigvals))
        is_pd = bool(min_eig > args.eig_tol)
        born_ok, born_failed = born_checks(csym, cs)
        is_stable = bool(is_pd and (born_ok if born_ok is not None else True))

        stats["pd_pass" if is_pd else "pd_fail"] += 1
        if born_ok is None:
            stats["born_na"] += 1
        else:
            stats["born_pass" if born_ok else "born_fail"] += 1
        stats["stable_pass" if is_stable else "stable_fail"] += 1

        by_cs[str(cs)]["n"] += 1
        by_cs[str(cs)]["stable_pass" if is_stable else "stable_fail"] += 1

        rows.append(
            {
                "material_id": mid,
                "crystal_system": cs,
                "min_eigenvalue": min_eig,
                "is_pd": is_pd,
                "born_applicable": born_ok is not None,
                "born_pass": bool(born_ok) if born_ok is not None else None,
                "born_failed_checks": born_failed,
                "is_stable": is_stable,
            }
        )

    n = stats["with_voigt"]
    print("=" * 70)
    print("ELASTIC STABILITY REPORT")
    print("=" * 70)
    print(f"Input: {args.in_json}")
    if args.denorm_stats_npz:
        print(f"Mode: de-normalized via {args.denorm_stats_npz}")
    else:
        print("Mode: raw values from input JSON")
    print(f"Total materials: {len(mats)}")
    print(f"With Voigt-21: {n}")
    print(f"Missing Voigt-21: {stats['missing_voigt']}")
    if n > 0:
        print(f"PD pass rate: {100.0 * stats['pd_pass'] / n:.2f}%")
        born_app = stats["born_pass"] + stats["born_fail"]
        if born_app > 0:
            print(f"Born pass rate (applicable systems): {100.0 * stats['born_pass'] / born_app:.2f}%")
        else:
            print("Born pass rate (applicable systems): N/A")
        print(f"Combined stability pass rate: {100.0 * stats['stable_pass'] / n:.2f}%")

    print("\nPer crystal system:")
    for cs, c in sorted(by_cs.items(), key=lambda kv: kv[1]["n"], reverse=True):
        denom = c["n"]
        rate = 100.0 * c["stable_pass"] / max(1, denom)
        print(f"  {cs or 'unknown'}: n={denom}, stable={rate:.2f}%")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()
