# Methodology

FuelCast provides 173,986 five-minute source observations across CPS Triton, CPS Poseidon and OSS Ceto. Onboard flow meters provide total momentary fuel flow; GPS provides SOG; external products supply interpolated environmental variables. The data expose a sequence index rather than calendar timestamps. Separate vessel models avoid pooling incompatible fuel scales and operating profiles.

Rows without a valid sequence index, target or SOG are excluded with an audit trail. Difficult weather and operating conditions are retained. Missing environmental values are median-imputed inside each training pipeline. Engine, fuel, shaft-power and propeller responses are excluded. Wind, wave and current directions are encoded relative to bearing using sine and cosine components.

Each ordered vessel sequence is divided into 60% training, 20% validation and 20% final test regions, with a 288-index gap before each held-out region. Candidate models are a third-order speed polynomial, Random Forest and CPU XGBoost, with speed-only and environmental feature ablations. Lowest validation MAE selects the frozen vessel model; all candidates are reported once on the final block. Temporal stability is then assessed using three predefined expanding-window evaluations with 10%-length future windows beginning at 55%, 70% and 85% of each sequence.

Voyage duration is distance divided by SOG. Predicted total fuel is the modelled kg/s multiplied by seconds. A fixed-environment speed grid is restricted by training quantiles, a travel-time constraint and a training-fitted nearest-neighbour support threshold. The method estimates conditional model outcomes and does not identify a causal speed response.
