# Multi-vessel verification report

Executed 2026-11-02 against FuelCast revision `eb6a6ec011c1c9a2cbce21459e22be4c77ef84dd`.

| Vessel | Type | Source rows | Columns | Retained rows | SOG availability | STW availability |
|---|---|---:|---:|---:|---|---|
| CPS Triton | Cruise passenger ship | 25,351 | 62 | 25,338 | 10 missing in source | Present but one unique nonmissing value |
| CPS Poseidon | Cruise passenger ship | 105,422 | 65 | 103,148 | 2,274 missing in source | Present with 4,703 unique values |
| OSS Ceto | Offshore supply ship | 43,213 | 46 | 43,197 | 9 missing in source | Absent |

All files contain the common target, SOG, sequence index, bearing and environmental fields required by the vessel-specific pipeline. Target values are available for every source row. All three files have no duplicate nonmissing sequence indices. File SHA-256 values are stored in each vessel metadata file and enforced by `scripts/download_data.py`.

The common experiment uses SOG because STW cannot be compared consistently across the three files. This choice supports a voyage-duration calculation but does not turn SOG into a hydrodynamic resistance variable.

## Cleaning audit

- CPS Triton: removed four rows without sequence index and nine additional rows without SOG.
- CPS Poseidon: removed 2,274 rows without SOG.
- OSS Ceto: removed eight rows without sequence index and eight additional rows without SOG.
- No target-dependent outlier filter was applied.
- Environmental gaps remain visible in the audit and are imputed only inside training pipelines.
- Each chronological split excludes 576 observations in two embargo regions.

## Operating-profile proxy

Speed-based regime labels are descriptive proxies. Triton is 73.1% transit, Poseidon 59.3% transit, and Ceto 83.5% near-stationary. These differences mean that model errors and R² values should be interpreted within each vessel rather than as a simple vessel ranking.

## Licence

The repository declares CC BY-NC-ND 4.0. Raw parquet is excluded from version control. Preserve source attribution and review licence terms before redistributing data or derivatives.
