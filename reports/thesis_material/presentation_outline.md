# Presentation outline and speaking cues

1. **Title** — Project title, student/program and one-sentence contribution.
2. **Why fuel efficiency matters** — Cost, emissions and operational decisions; avoid unsupported global statistics.
3. **Problem statement** — Predict total fuel flow and compare supported constant-SOG voyage alternatives.
4. **Practical gap** — Speed-only curves miss context; ML predictions alone do not create a valid optimizer.
5. **Objectives** — Data verification, prospective prediction, model comparison, constrained grid and dashboard.
6. **FuelCast data** — Three vessels, 173,986 source rows, 5-minute sequences, kg/s target; weather is external/interpolated.
7. **Data quality and vessel profiles** — 171,683 retained; inconsistent STW availability; Ceto is predominantly near-stationary.
8. **System architecture** — Show the flow diagram; emphasize frozen test block and shared inference logic.
9. **Models and features** — Cubic baseline, RF, XGBoost; speed-only and environmental ablations; no engine-response leakage.
10. **Evaluation** — 60/20/20 chronological regions plus one-day embargoes; validation selects, test audits once.
11. **Vessel-specific results** — Show three selected models and explain why Poseidon selects the cubic baseline.
12. **Repeated-window reliability** — Environmental XGBoost has the best mean MAE for all vessels; variability is material for Poseidon and Ceto.
13. **Speed comparison** — Explain `fuel = predicted rate × distance/SOG`, support checks and infeasible handling.
14. **Dashboard demonstration** — Data Explorer → Prediction → Speed Optimization → Evaluation. Deliberately enter an unsupported scenario to show the warning.
15. **Conclusion and future work** — 14.3% test-MAE improvement over speed-only XGBoost; savings remain exploratory; independent voyage/sea-trial validation is next.

## Likely panel questions

- **Why chronological splitting?** Adjacent five-minute records are autocorrelated; a random split would allow near-neighbours into both sets.
- **Why not use STW?** The inspected channel has only one nonmissing value, so it carries no usable variation.
- **Why can faster speed use less voyage fuel?** The target includes all consumers and a shorter voyage reduces time-dependent auxiliary/hotel consumption; the model is associational, so this result requires validation.
- **Is 27.25% a proven saving?** No. It is one supported conditional model estimate and is explicitly labelled exploratory.
- **Why did Poseidon select a cubic model?** Its fixed validation period favored the baseline. Repeated windows favor environmental XGBoost on average, demonstrating temporal variability rather than a universal winner.
- **Are these cross-vessel models?** No. They are separately fitted vessel-specific models using a common feature and evaluation protocol.
- **What is the main weakness?** Temporal variability, low-speed generalisation and missing causal operating context.
- **Why XGBoost?** It provided the lowest validation MAE with modest CPU cost and captured nonlinear interactions.
