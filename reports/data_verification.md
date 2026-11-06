# Milestone 1: verified dataset

Inspection executed on 2026-11-02 using pandas and PyArrow. Full column types, missing counts, numeric ranges and SHA-256 are in `metrics/inspection.json`.

- Source: https://huggingface.co/datasets/krohnedigital/FuelCast
- Repository revision: eb6a6ec011c1c9a2cbce21459e22be4c77ef84dd
- Selected file: CPS_Triton.parquet, 25,351 rows, 62 columns.
- Target: Consumer_Total_MomentaryFuel, mass flow in kg/s, including all consumers. Not cumulative voyage fuel.
- Speed: Ship_SpeedOverGround, m/s. Ship_SpeedThroughWater exists but all its nonmissing values are approximately 0.514444 m/s; exclude this uninformative channel.
- Time: numeric `index`, documented five-minute sequence position. No actual calendar timestamp should be inferred. Inspect and resolve duplicate indices before partitioning.
- Vessel identity comes from the file, not a fabricated column.
- Weather_WindSpeed10M, Weather_WaveHeight and Weather_OceanCurrentVelocity exist. Wind units in the card contain a typo; the accompanying paper Table 2 specifies m/s.
- Source documentation identifies onboard flow measurements and externally derived, interpolated environmental inputs; these are not onboard weather sensor observations.
- Idle and moving observations coexist: SOG quartiles approximately 0.741, 7.110 and 7.439 m/s. Operating regimes must be explicitly defined as speed proxies, not labelled manoeuvres.
- Raw data is preserved. No modelling or performance claims have been made.

## Licence and attribution
The repository declares CC BY-NC-ND 4.0. Preserve attribution and the original source; do not put the raw or processed datasets in Git. This report does not determine redistribution rights for derivatives.

## References
Viga, Mueck, Löser and Weis, *FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption*, https://arxiv.org/abs/2510.08217 . Dataset card archived in `source/dataset_card.md`.

## Engineering implication
Use SOG for voyage duration and a SOG-based predictive model. Do not treat it as STW or infer propulsive resistance directly. Current direction conventions and missing load/route context constrain counterfactual interpretation; present the voyage module as exploratory supported speed comparison.
