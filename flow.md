# System flow

```mermaid
flowchart TD
  A[Pinned CPS Triton parquet] --> B[Validate required schema]
  B --> C[Coerce numeric values and audit exclusions]
  C --> D[Sort five-minute sequence index]
  D --> E[60% train]
  D --> F[20% validation after 288-index embargo]
  D --> G[20% final test after 288-index embargo]
  E --> H[Feature construction and training-only imputation]
  H --> I[Cubic / RF / XGBoost candidates]
  F --> J[Select lowest validation MAE]
  I --> J
  J --> K[Freeze selected training-fitted pipeline]
  K --> L[Single final-test audit]
  K --> M[Prediction with empirical support check]
  M --> N[Feasible speed grid]
  N --> O[Duration and total-fuel calculation]
  L --> P[Saved tables, figures and dashboard]
  O --> P
```

The loader validates the common real schema separately for CPS Triton, CPS Poseidon and OSS Ceto. Cleaning records every exclusion. No quantile outlier filter is used. Environmental gaps remain in processed data and are imputed inside model pipelines using vessel-specific training medians.

Prediction inputs contain ground speed, environmental scalar variables and relative-direction sine/cosine terms. They exclude fuel, engine speed, shaft power and propeller variables. Model selection uses validation MAE. Test metrics are descriptive evidence for one future block and never feed selection.

Each vessel receives an independent inference bundle containing its selected pipeline, feature order, validation error sensitivity, speed envelope and training-only nearest-neighbour support model. Three expanding-window tests additionally compare the cubic baseline and both XGBoost feature sets at 55%, 70% and 85% sequence positions. The optimizer duplicates one observed vessel-specific environmental snapshot over a speed grid, predicts fuel rate, rejects unsupported combinations, applies the arrival-time constraint, then minimizes calculated voyage fuel among feasible grid points. The dashboard calls this shared logic and never trains a model.
