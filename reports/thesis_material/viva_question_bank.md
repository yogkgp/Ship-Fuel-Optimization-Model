# Viva question bank

## Dataset and preprocessing

**Why did you choose FuelCast?**  
It provides real, public five-minute operational data for three anonymised vessels together with fuel-flow and environmental variables. This supports reproducible vessel-specific prediction without presenting synthetic values as measurements.

**What exactly is the target?**  
`Consumer_Total_MomentaryFuel`, the measured total momentary mass flow of all recorded consumers in kg/s. It is neither cumulative fuel nor propulsion-only fuel.

**Why did you use speed over ground?**  
It is the only comparable speed channel across all vessels. Triton’s STW is constant, Poseidon’s varies, and Ceto has no STW field. SOG supports duration calculation, but I do not interpret it as hydrodynamic speed through water.

**How did you handle missing values?**  
Rows without sequence index, target or SOG were unusable and audited as exclusions. Environmental values were median-imputed inside each training pipeline, so held-out observations did not affect the imputation values.

**Did you remove outliers to improve accuracy?**  
No quantile or target-dependent outlier filter was used. Impossible negative or invalid directional inputs are treated as missing, while difficult real operating conditions remain in the data.

## Evaluation

**Why not use a random train-test split?**  
Adjacent five-minute samples are autocorrelated. A random split would place near-neighbour observations in training and testing and overstate future-period performance.

**What is the purpose of the 288-index gap?**  
It creates a one-day embargo before validation and test regions, reducing direct leakage from highly adjacent observations.

**Why do you report both a fixed split and repeated windows?**  
The fixed validation block provides an auditable model-selection rule and a final test. The three additional windows show whether the ranking and error remain stable over different future periods.

**Why did Poseidon select the cubic baseline?**  
The fixed validation period favored the cubic model. Environmental XGBoost had lower mean MAE across three other windows, which indicates temporal variability. I retain the predefined validation decision rather than selecting a model after seeing test results.

**Can you compare R² directly across vessels?**  
Only cautiously. Fuel scale, variance and operating profile differ. Ceto is mostly near-stationary, while Triton is mostly transit. MAE in physical units and within-vessel comparisons are more informative.

## Models and interpretation

**Why Random Forest and XGBoost?**  
Both capture nonlinear effects and interactions in tabular data, run efficiently on a CPU, and provide useful baselines against a cubic speed relationship. XGBoost delivered the best average repeated-window MAE.

**Do environmental features cause lower fuel consumption?**  
The models show predictive association. Route, season and vessel operation can correlate with weather, so the experiment does not isolate causal effects.

**How did you avoid target leakage?**  
I excluded all fuel variables and downstream engine, shaft-power, propeller-power and rotation-speed measurements. Inputs contain SOG and environmental context available for scenario analysis.

**What does permutation importance mean here?**  
It measures how much held-out MAE increases when a feature is shuffled. It describes predictive dependence and can distribute importance unpredictably among correlated variables.

## Uncertainty

**How are prediction intervals calculated?**  
Absolute residuals from the validation block calibrate finite-sample split-conformal radii. Separate radii are used for near-stationary, low-speed and transit regimes when at least 100 calibration observations are available.

**Does a nominal 90% interval guarantee 90% future coverage?**  
No. Split conformal coverage assumes exchangeability. Observed coverage is 89.8% for Triton, 93.6% for Poseidon and 79.4% for Ceto. Ceto’s result demonstrates temporal distribution shift.

**Why keep the Ceto interval if coverage is poor?**  
The diagnostic is useful because it identifies where the model is unreliable. The dashboard exposes undercoverage rather than giving a false confidence guarantee.

## Speed decision support

**What is the optimization objective?**  
For each supported candidate SOG, estimated voyage fuel equals predicted total fuel rate multiplied by distance divided by SOG. The grid minimizes this estimate subject to the time limit and support constraints.

**Why can a faster speed sometimes reduce voyage fuel?**  
The target includes auxiliary and hotel loads. A shorter voyage can reduce time-dependent consumption. The associational model may also reflect route and operating correlations, so such results require operational validation.

**Is the recommended speed a physical optimum?**  
No. It is the minimum on a finite supported model grid under a fixed environmental snapshot. It excludes acceleration, routing, forecast changes and causal verification.

**How do support checks work?**  
Candidates must remain inside training feature ranges and within a threshold based on the tenth nearest training neighbour after scaling. This reduces extrapolation but cannot prove operational feasibility.

## Limitations and future work

**What is the strongest limitation?**  
The project uses observational, vessel-specific data without voyage labels or sea-trial interventions. Temporal shift is visible, particularly for Ceto.

**What would you do next?**  
Add voyage segmentation, reliable STW and draft, separate propulsion and auxiliary fuel, incorporate forecast uncertainty, test transfer to an unseen vessel and validate recommendations against measured voyages.
