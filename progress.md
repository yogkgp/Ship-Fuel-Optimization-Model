# Project progress

Updated: 2026-11-02 (Asia/Kolkata)

## Completed
- Inspected empty workspace and hardware: Intel i3-1005G1, 11.8 GB RAM, Windows 11.
- Verified FuelCast repository and saved dataset card and repository metadata.
- Downloaded CPS_Triton.parquet (3,154,132 bytes); raw file remains unchanged.
- Verified documented units, provenance, five-minute index and restrictive source licence.

## In progress
- Final dashboard smoke test and presentation handoff.

## Pending
- Optional multi-vessel extension and independent voyage validation; outside the focused first version.

## Blockers and limitations
- Only 2.9 GB RAM was free at initial inspection; use bounded models and two CPU threads.
- No calendar timestamps: use elapsed sample time, never invent dates.
- Counterfactual speed changes cannot establish causal savings.

## Checks executed
- Python 3.12.10 verified; dataset download completed. No model tests yet.

Next action: launch and rehearse the Streamlit dashboard.

## Milestone 1 inspection complete
- Executed scripts/inspect_data.py: 25,351 rows, 62 columns; full JSON report saved.
- No exact duplicate rows or negative targets. Four missing sequence indices; ten missing SOG values. Sequence is not monotonic in source order, so explicit sorting and missing-index rules are required.
- STW has one unique nonmissing value and must be excluded.
- Created reports/data_verification.md and decisions.md.
- Dependencies may still be installing in background; no training or tests yet.
- Next: inspect installation status, install requirements.txt to enforce bounds, implement cleaning with sorted valid indices, and continue milestone 2.

## Milestones 2–6 complete
- Cleaned 25,338 observations. Chronological split: 14,914 train, 4,779 validation and 5,069 final test; 576 embargo observations.
- Generated 18 EDA, diagnostic and engineering figures.
- Evaluated cubic baseline, Random Forest and XGBoost with speed/environment ablations.
- Validation selected XGBoost with environmental inputs. Final-test MAE 0.01837 kg/s, RMSE 0.02366 kg/s and R² 0.834.
- Implemented bounded voyage comparison with support, deadline and infeasibility checks.
- Created six-page Streamlit dashboard, README, flow, decision log, experiment log and thesis/presentation material.
- Automated tests: 15 passed on 2026-11-02.
- Known limitation: low-speed final-test R² is −0.057; voyage savings are conditional estimates, not verified savings.

## Three-vessel extension complete
- Downloaded and SHA-256 verified CPS Poseidon and OSS Ceto at the pinned FuelCast revision.
- Retained 25,338 Triton, 103,148 Poseidon and 43,197 Ceto observations after explicit cleaning.
- Trained five configurations separately for each vessel; runtime approximately three minutes on this laptop.
- Executed three expanding future-window tests for cubic, speed-only XGBoost and environmental XGBoost.
- Environmental XGBoost had the lowest three-window mean MAE for every vessel, with substantial Poseidon/Ceto variability.
- Added vessel selection and repeated-window tables to the dashboard.
- Added three cross-vessel figures and synchronized experiment/thesis documentation.
- Automated tests: 33 passed on 2026-11-02, including every dashboard page, vessel-specific intervals and all three vessel selections.

## Reliability and presentation extension complete
- Added finite-sample split-conformal prediction intervals calibrated by vessel and speed regime.
- Measured final-test coverage: Triton 89.8%, Poseidon 93.6%, Ceto 79.4% for nominal 90% intervals.
- Exposed coverage diagnostics and warnings in the dashboard.
- Created a 15-slide live presentation and reviewed all rendered slides at full size.
- Created a viva question bank covering data, evaluation, models, uncertainty, optimization and limitations.

## Saved checkpoint
- All workspace files and generated artifacts verified on disk on 2026-11-02.
- Resume instructions are in `RESUME_TOMORROW.md`.
- Live presentation artifact reference: `artifact_LeKGvVxxKJP_L67A5gl4N5VceDvs4Cv3`.
