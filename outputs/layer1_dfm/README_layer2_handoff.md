# Layer 2 handoff

## Primary entry point
- Primary training table: `layer2_residual_design.parquet`
- CSV fallback: `layer2_residual_design.csv`
- Feature manifest: `layer2_feature_manifest.csv`
- Data contract: `layer2_data_contract.json`

## Row grain
One row per `(vintage_period, target_quarter)`.

## Primary key
- `vintage_period`
- `target_quarter`

## Primary target
- `dfm_residual_third_release`

## Robustness targets
- `dfm_residual_latest_rtdsm`
- `dfm_residual_gdpplus`

## Train sample filter
- Use rows where `primary_target_available == True`

## Required date semantics
- `vintage_period` must be treated as monthly period
- `target_quarter` must be treated as quarterly period
- preserve first-day-of-month / first-day-of-quarter semantics exactly
- do not convert to month-end timestamps

## Feature policy
- Build `X` only from columns where `included_in_training_matrix == True` in `layer2_feature_manifest.csv`
- Do not auto-select all numeric columns

## Baseline recommendation
Exclude from the first baseline run:
- `news_signed__quarterly_target_history`
- `news_abs__quarterly_target_history`

These may be tested later in robustness runs.

## Audit-only / forbidden fields
Respect `forbidden_feature_columns` and `audit_only_fields` in `layer2_data_contract.json`.

## Evaluation rule
Use time-aware pseudo-real-time splits and fit preprocessing only inside each training fold.