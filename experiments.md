# Executed experiments

## E01 — Dataset verification

Objective: verify provenance, schema, units and data quality. Dataset: FuelCast CPS Triton, repository revision `eb6a6ec…`, SHA-256 recorded in model metadata. Result: 25,351 source rows and 62 columns. Target is total momentary fuel flow in kg/s; SOG is m/s. Four rows lack sequence indices, ten lack SOG, and the constant STW channel is unusable. Full audit: `reports/metrics/inspection.json`.

## E02 — Chronological prediction

Features: speed only for baseline/ablations; speed plus environmental variables and relative directions for full models. Split: 14,914 train / 4,779 validation / 5,069 test, with 288-index embargoes. Configuration: cubic polynomial; RF 120 trees, depth 16, minimum leaf 12; XGBoost 240 trees, depth 4, learning rate 0.04. Two CPU threads, seed 42.

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Cubic baseline | 0.02423 | 0.02223 | 0.02880 | 0.754 |
| Random Forest, speed | 0.02223 | 0.02158 | 0.02854 | 0.758 |
| XGBoost, speed | 0.02213 | 0.02144 | 0.02831 | 0.762 |
| Random Forest, environment | 0.01888 | 0.01933 | 0.02452 | 0.822 |
| **XGBoost, environment** | **0.01853** | **0.01837** | **0.02366** | **0.834** |

Units for MAE/RMSE are kg/s. XGBoost with environmental inputs was selected before test evaluation. Its test MAE is 14.3% lower than speed-only XGBoost and 17.4% lower than the cubic baseline. This supports predictive value from environmental context in this split, but does not isolate causal weather effects.

Regime analysis: near-stationary MAE 0.02107; low-speed MAE 0.02566 and R² −0.057; transit MAE 0.01728. Low speed is the least reliable proxy regime. Wave-band MAE ranges from 0.01437 to 0.01868 kg/s, but sample size and operating mix differ.

## E03 — Supported voyage comparison

Method: 151-point SOG grid bounded to 2.0–8.19 m/s, empirical marginal and local-neighbour support, constant environment, exact distance/time unit conversions. Three examples use actual training rows rather than fabricated scenarios. For a 100 nm median-wave example, the model grid changes the reference from 8.32 kn / 7.42 t / 12.02 h to 14.40 kn / 5.40 t / 6.95 h, an estimated 27.25% difference. This counterintuitive result is possible because total fuel includes non-propulsive consumers and because the low-speed reference is a conditional association. It must be described as an illustrative model estimate, not a verified efficiency recommendation.

For that example, a 6-hour limit is infeasible within supported candidates; limits of 8 hours or more select the same supported grid point. Other observed examples produced 0.39% and 2.48% estimated differences, showing strong scenario dependence.

Next step for research-grade optimization: validate on independent voyages using reliable speed through water, propulsion/hotel-load separation, draft and forecast uncertainty, followed by operational or sea-trial validation.

## E04 — Three vessel-specific models

All three pinned FuelCast files passed the required common schema. CPS Triton retained 25,338 rows, CPS Poseidon 103,148, and OSS Ceto 43,197. The same five candidates and chronological protocol were applied separately.

| Vessel | Validation-selected model | Final-test MAE | RMSE | R² |
|---|---|---:|---:|---:|
| CPS Triton | XGBoost + environment | 0.01837 | 0.02366 | 0.834 |
| CPS Poseidon | Cubic baseline | 0.07510 | 0.11387 | 0.950 |
| OSS Ceto | XGBoost + environment | 0.04570 | 0.06625 | 0.579 |

Environmental XGBoost reduced final-test MAE against the cubic baseline by 17.4% for Triton and 32.8% for Ceto. On Poseidon it was 8.1% worse on that fixed final block. The result rejects a simplistic claim that added environmental features always improve every time split.

## E05 — Repeated chronological windows

Three predefined 10%-length future windows start after 55%, 70% and 85% of each vessel sequence. Training expands from the beginning and stops one 288-index embargo before each window. Mean MAE ± standard deviation:

| Vessel | Cubic baseline | XGBoost speed | XGBoost environment |
|---|---:|---:|---:|
| CPS Triton | 0.02392 ± 0.00039 | 0.02246 ± 0.00039 | **0.01829 ± 0.00183** |
| CPS Poseidon | 0.07363 ± 0.01119 | 0.07356 ± 0.02078 | **0.06339 ± 0.02450** |
| OSS Ceto | 0.05282 ± 0.00908 | 0.05806 ± 0.00828 | **0.04437 ± 0.01645** |

Environmental XGBoost has the lowest mean MAE for all three vessels, but Poseidon and Ceto have large variability. Ceto’s R² also changes sharply across windows because its dynamic-positioning/near-stationary operating mix is heterogeneous. Complete fold results are in `reports/metrics/repeated_temporal_evaluation.csv`.

## E06 — Regime-calibrated uncertainty

Finite-sample split-conformal radii use absolute residuals from each vessel's validation block. Near-stationary, low-speed and transit regimes receive separate radii when the calibration group contains at least 100 observations; smaller groups fall back to the global radius. Nominal coverage is 90%.

| Vessel | Overall observed coverage | Mean interval width |
|---|---:|---:|
| CPS Triton | 89.8% | 0.0772 kg/s |
| CPS Poseidon | 93.6% | 0.4059 kg/s |
| OSS Ceto | 79.4% | 0.1466 kg/s |

Triton is close to nominal coverage and Poseidon is conservative overall. Ceto undercovers substantially, especially near-stationary observations (78.8%). This demonstrates that chronological distribution shift violates the exchangeability assumption used for conformal coverage guarantees. Intervals are therefore presented as calibrated diagnostics with observed coverage, not guaranteed confidence bounds.
