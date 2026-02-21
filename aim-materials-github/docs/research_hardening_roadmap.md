# AIM Materials: Reliability-to-Publishability Roadmap

## 0) Immediate fixes (implemented in this sprint)
- Wire `target_num_elements` from dataset -> batch -> model forward.
- Fix undefined `target_num_elements` usage in `AimMultiModalModel.forward`.
- Make composition-diversity prior configurable (`max_elements_loss`).
- Add generation options for composition control:
  - `--target_num_elements`
  - `--enforce_exact_num_elements`
- Align MP export Voigt unpacking order with preprocessing order.
- Add `evaluate_elastic_stability.py` to compute PD/Born stability pass rates.

## 1) Data integrity gates (week 1)
- Regenerate training JSON from `processed_filtered_mp.py` so `target_num_elements`/`nelements` are explicit in samples.
- Freeze dataset snapshot and record:
  - source file hashes,
  - split manifest (`train/val/test` IDs),
  - normalization stats version.
- Enforce schema validation on every dataset build.

## 2) Model correctness gates (week 1-2)
- Add unit tests for:
  - Voigt-21 pack/unpack order consistency,
  - conditioning tensor shapes and batch behavior,
  - loss finiteness under random stress tests.
- Add regression tests on fixed mini-benchmark set with expected metric ranges.

## 3) Physics constraints (week 2)
- Add explicit stability-aware loss terms:
  - positive-definite stiffness penalty,
  - Born-criteria violations per crystal system.
- Add post-inference physical validation:
  - PD check,
  - Born checks,
  - minimum interatomic distance and lattice sanity.

## 4) Generation validity and controllability (week 2-3)
- Train and report generation under controlled composition buckets (2/3/4).
- Add hard filters for:
  - composition validity,
  - charge-neutrality heuristics,
  - duplicate/near-duplicate structures.
- Track conditional success rates:
  - requested `nelements` vs achieved `nelements`.

## 5) Evaluation protocol for publication (week 3-4)
- Define fixed evaluation suite:
  - scalar MAE/RMSE/R2 (8 targets),
  - Voigt MAE + symmetry consistency,
  - PD/Born pass rates,
  - novelty/diversity against MP reference.
- Produce ablations:
  - without/with conditioning,
  - without/with physics losses,
  - without/with generation hard constraints.
- Prepare reproducibility package:
  - exact commands,
  - seeds,
  - configs,
  - checkpoints and logs.

## 6) Publishability bar (Nature-style expectations)
- Clear scientific claim with uncertainty quantification.
- Strong baselines and ablation evidence.
- External validation (holdout domains or independent datasets).
- Candidate shortlist with physically valid, novel, and high-performing materials.
