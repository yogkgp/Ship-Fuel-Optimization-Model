# Engineering decisions

## 2026-11-02 — One-vessel first implementation
Decision: start with CPS Triton. Alternatives: Poseidon or all three vessels. Reason: smallest published file and manageable laptop training. Consequence: no cross-vessel generalisation claims.

## 2026-11-02 — Python 3.12 isolated environment
Decision: use `.venv` with installed Python 3.12.10. Alternative: default Python 3.14. Reason: conservative project dependency environment independent of global packages. Consequence: separate dependency installation required.

## 2026-11-02 — Exclude constant water-speed channel
Decision: exclude Ship_SpeedThroughWater after actual inspection found one unique nonmissing value. Alternative: use both speeds. Reason: no useful variation and physically suspect channel. Consequence: SOG scenario comparison cannot establish resistance or causal savings.

## 2026-11-02 — Prospective evaluation
Decision: planned chronological 60/20/20 partitions with one-day embargoes (288 sample indices). Alternative: random split. Reason: adjacent observations are autocorrelated. Consequence: fewer training/evaluation records; actual split counts pending implementation.

## 2026-11-02 — Freeze the validation winner
Decision: select the training-fitted candidate with lowest validation MAE and evaluate that frozen pipeline on the final block without refitting. Alternatives: retrain on train+validation or tune against test results. Reason: keep the final block independent and make the selection rule auditable. Consequence: the deployed model does not use validation observations for fitting.

## 2026-11-02 — XGBoost with environmental inputs
Decision: use environmental XGBoost as the demonstration model after it achieved the lowest validation MAE (0.01853 kg/s). Alternatives: cubic baseline, speed-only RF/XGBoost and environmental RF. Reason: predefined validation criterion. Consequence: final-test MAE is 0.01837 kg/s; this remains one-vessel, one-period evidence.

## 2026-11-02 — Grid search and empirical support
Decision: use a transparent 151-point grid and require both marginal training bounds and local nearest-neighbour support. Alternative: continuous numerical optimization or unrestricted model extrapolation. Reason: inspectable feasibility and explicit rejection of sparse combinations. Consequence: the returned point is a supported grid minimum, not proof of a physical optimum.

## 2026-11-02 — Label voyage results exploratory
Decision: report conditional estimates rather than optimized/verified savings. Alternative: treat input edits as causal interventions. Reason: SOG is not STW, total fuel includes all consumers, and speed/context relationships are observational. Consequence: operational deployment requires independent voyage and sea-trial validation.

## 2026-11-02 — Separate model per vessel
Decision: fit independent CPS Triton, CPS Poseidon and OSS Ceto pipelines with the same common SOG/environment features. Alternative: pool vessel identities or train a transfer model. Reason: vessel scale and operating profiles differ strongly, and three vessels are insufficient to establish fleet transfer. Consequence: results support within-vessel prediction only.

## 2026-11-02 — Three expanding future windows
Decision: supplement the fixed selection/test protocol with predefined windows starting at 55%, 70% and 85% of each sequence. Alternative: random cross-validation or exhaustive five-fold tuning. Reason: assess temporal stability at modest CPU cost while preserving order and embargoes. Consequence: folds share earlier training history and should not be treated as independent voyages.
