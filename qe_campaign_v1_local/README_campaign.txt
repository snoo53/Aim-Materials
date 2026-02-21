QE Campaign Package (Local/Open-source)
=======================================
1) Place UPF pseudo files under the pseudo directory referenced by qe_*.in.
2) Relax: run ./run_relax.sh [pw.x] [nproc] or .\run_relax.ps1.
3) SCF: run ./run_scf.sh [pw.x] [nproc].
4) Elastic (finite-strain stress): run ./run_elastic.sh [pw.x] [nproc].
5) Analyze outputs with analyze_qe_campaign_results.py.

Notes:
- Elastic constants are fitted from stress response of +/- strain_amp perturbations.
- Uses engineering shear convention in Voigt notation.
- If your QE stress sign appears inverted, use --qe_stress_sign in analyzer.
