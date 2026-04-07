# Real-Time GDP Nowcasting with a Hybrid Dynamic Factor Model and Machine Learning Residual Correction

## Project purpose

This project studies **real-time U.S. GDP nowcasting** using a **two-layer hybrid pipeline**:

1. **Layer 1: Dynamic Factor Model (DFM)** in state-space form produces a structurally coherent real-time nowcast of quarterly real GDP growth.
2. **Layer 2: Machine Learning (ML) residual correction** does **not** forecast GDP directly. Instead, it learns the remaining predictable error left by the DFM and adds that correction back to the DFM baseline.

The core research question is:

> Can machine learning extract **predictable nonlinear structure** from the residuals of a rigorously specified **real-time dynamic factor model**, while preserving the vintage-aware, ragged-edge-capable, and interpretable structure of a state-space nowcasting system?

This is **not** a generic forecasting project and **not** a static “latest-data” GDP prediction exercise. It is a **real-time, vintage-aware nowcasting system** built under the rule:

> **At each forecast date, the model may use only the information that truly existed at that date.**

That principle is often summarized as **what was known when**.

---

## What this repository is trying to do

The repository is designed to build a **publication-grade, reproducible nowcasting pipeline** that:

- uses **real-time vintages** rather than revised latest histories,
- handles **mixed-frequency data** correctly,
- treats the **ragged edge** inside the state-space system rather than through ad hoc external imputation,
- keeps the DFM as the **econometric backbone**,
- and uses ML only as a **disciplined residual-correction layer**.

The intended scientific contribution is not “ML versus econometrics” in the abstract. The sharper identification logic is:

- estimate a real-time DFM on the correct information set,
- produce a DFM-only nowcast,
- define the DFM error in pseudo real time,
- then test whether ML can learn residual structure that remains **after** the common linear macroeconomic state has already been extracted.

If the hybrid model outperforms the DFM-only benchmark under the **same real-time information set**, the gain must come from **nonlinear residual predictability**, not merely from adding more indicators.

---

## Core model idea

### Layer 1: DFM backbone

The first layer is a **mixed-frequency dynamic factor model** written in **state-space form**.

Its role is to:

- compress a large monthly macro panel into a small latent macro state,
- handle arbitrary missingness caused by publication delays,
- bridge monthly information to quarterly GDP,
- and support model-consistent **news decomposition** of nowcast revisions.

Conceptually:

- many monthly indicators co-move because they reflect common macroeconomic forces,
- the DFM extracts those latent forces,
- and quarterly GDP is linked to the monthly state through a mixed-frequency measurement design.

This layer is the structural, interpretable, and ragged-edge-capable backbone of the whole project.

### Layer 2: ML residual correction

The second layer is **not a standalone GDP predictor**.

Instead, it learns the DFM residual:

- DFM baseline nowcast: `g_hat_DFM(v, q)`
- realized target: `g_truth(q)`
- residual: `e(v, q) = g_truth(q) - g_hat_DFM(v, q)`

The ML model then learns a mapping from vintage-consistent real-time features to that residual:

- factors and state summaries,
- block-level news,
- ragged-edge / coverage descriptors,
- release-flow descriptors,
- and other low-dimensional objects produced by Layer 1.

The final hybrid nowcast is:

`g_hat_HYB(v, q) = g_hat_DFM(v, q) + e_hat(v, q)`

So the intended division of labor is:

- **DFM** captures broad common macro structure,
- **ML** captures nonlinear or interaction-driven structure left in the residual.

---

## Time semantics and notation

This project depends critically on **correct time semantics**. Misinterpreting timestamps is one of the easiest ways to corrupt the design.

### Indices

- `q`: target quarter
- `m`: month
- `v`: month-end vintage / information date
- `I_v`: information set available at vintage `v`

### Target variable

The target is **quarter-over-quarter real GDP growth in annualized percentage points**:

`g_q = 400 * [log(Y_q) - log(Y_{q-1})]`

where `Y_q` is real GDP in quarter `q`.

### Vintage meaning

A vintage `v` does **not** mean that every series has data through that same calendar month.

Instead:

- `v` labels the **information set available by the end of that month**,
- but different series may be available through different last-observed months because of release lags.

So within the same vintage:

- some indicators may already include the current month,
- some may only be available through the previous month,
- and some may lag even further.

This changing missing-data pattern is the **ragged edge**.

### Repository timestamp convention

To keep merges, contracts, and reproducibility stable, this repository uses:

- **first day of month** to encode monthly periods,
- **first day of quarter** to encode quarterly periods.

Examples:

- monthly period `2026-03` is encoded as `2026-03-01`
- quarter `2026Q1` is encoded as `2026-01-01`

Important:

- these timestamps are **period identifiers**, not literal “release-day” timestamps,
- and `vintage_period = 2026-03-01` should be read as:
  - **the information set available by the end of March 2026**

### Within-quarter forecast origins

For each target quarter `q`, the project generates **three nowcasts**, one at the end of each month of that quarter.

Denote this by:

- `tau(v, q) in {1, 2, 3}`

Example for `2026Q1`:

- end of January 2026 -> `tau = 1`
- end of February 2026 -> `tau = 2`
- end of March 2026 -> `tau = 3`

This design is native to the real-time nowcasting problem and tracks how the nowcast evolves as information accumulates within the quarter.

---

## Data architecture

The empirical design combines a **quarterly target-vintage system** with a **large monthly predictor-vintage system**.

### Target side

The target side should be built from **Philadelphia Fed RTDSM / ROUTPUT**.

Use three distinct target objects:

1. **Vintage target history**
   - GDP history as it truly existed at each vintage `v`
   - used for real-time DFM estimation

2. **Main truth table**
   - primary scoring target = **third-release real GDP growth**
   - used for main forecast evaluation

3. **Robustness truth tables**
   - latest available RTDSM value
   - Philadelphia Fed GDPplus

This distinction is essential.

**Do not mix up:**
- the GDP history that the DFM is allowed to see at vintage `v`,
- and the later truth definition used to score forecasts.

### Predictor side

The main predictor information set comes from the **FRED-MD vintage archive**.

Additional data sources play narrower roles:

- **ALFRED / FRED API**:
  - audit release dates,
  - validate vintage timing,
  - check individual series,
  - not the main predictor flow

- **FRED-QD**:
  - supplementary robustness,
  - not the core Layer 1 input panel

- **Survey of Professional Forecasters (SPF)**:
  - optional external benchmark

### Why FRED-MD is central

FRED-MD is the main panel because it is:

- public,
- broad,
- standardized,
- transformation-code aware,
- and suitable for reproducible pseudo-real-time macro forecasting.

The project is built around the **monthly flow of information**, so FRED-MD is the right central predictor system.

---

## Real-time design rules

These rules are non-negotiable if the project is to remain methodologically valid.

### Rule 1: vintage integrity

At vintage `v`, every observation used by the model must be recoverable exactly as it existed at that month-end.

### Rule 2: no future contamination

The project must not use:

- future data values,
- future revisions,
- future standardization moments,
- or future availability patterns.

### Rule 3: truth is separate from vintage history

The GDP history used for DFM estimation at vintage `v` is **not the same object** as the truth used for later evaluation.

### Rule 4: ragged edge stays inside state-space

Missing current-quarter values caused by release lags must remain missing in the measurement system and be handled by the **Kalman filter / smoother**.

Do **not** fill the ragged edge with a separate external imputation model before fitting the DFM.

### Rule 5: ML must remain vintage-consistent

The residual learner may only use features that were actually available at the relevant pseudo-real-time decision date.

### Rule 6: no residual training without realized truth

A quarter whose truth is not yet available must **not** contribute a supervised residual target to Layer 2 training.

---

## Mixed-frequency and ragged-edge interpretation

Nowcasting quarterly GDP from monthly indicators is intrinsically a **mixed-frequency problem**.

The target is quarterly, while most predictors are monthly and arrive on different release schedules.

At a given vintage `v`:

- some indicators cover all months of the current quarter,
- some cover only one or two months,
- some have longer publication lags.

This creates the ragged edge.

The project does **not** treat this as a nuisance to be pre-filled away.

Instead, it treats ragged-edge missingness as an **endogenous real-time feature of the information set** and lets the state-space DFM absorb it through Kalman filtering/smoothing.

This is a core methodological choice and should not be changed casually.

---

## Predictor transformation and preprocessing

At each vintage `v`, predictor preprocessing should follow this order:

1. **freeze the predictor snapshot at vintage `v`**
2. **apply official FRED-MD transformation codes**
3. **standardize using only the estimation sample available at vintage `v`**

This means:

- no global full-sample scaling,
- no using future means or standard deviations,
- no static preprocessing pipeline reused across all vintages without recalculation.

This repository should preserve the moments used at each vintage so the exact transformed panel can be reproduced later.

---

## Panel design: full panel and stable subset

Because FRED-MD composition can evolve across vintages, the project should maintain two predictor definitions:

### 1. Full panel
Uses the actual series list available in each vintage file.

### 2. Stable subset
A fixed ex ante subset used for robustness, interpretability, and debugging.

This split is useful because:

- the full panel captures the richest real-time information set,
- the stable subset helps verify that results are not driven only by panel redesign or series churn.

---

## Economic block structure

To preserve interpretability and structure the DFM, predictors are grouped into six economic blocks:

1. real activity and income
2. labor market
3. housing and construction
4. demand, orders, and inventories
5. prices and inflation
6. financial conditions

This block structure is not cosmetic.

It matters because it affects:

- factor construction,
- news decomposition,
- coverage metrics,
- and interpretable ML feature groups in Layer 2.

A stable ex ante `series_to_block` mapping should therefore be maintained.

---

## Coverage and ragged-edge summary features

For each series `i`, target quarter `q`, and vintage `v`, define the set of observed months in that quarter.

From this, the project may construct:

- latest observed value in quarter,
- change from the previous available observation,
- partial-quarter average,
- months missing,
- block-level coverage statistics.

These objects summarize the **state of information arrival**, not missing-value imputations.

Examples of useful feature families for Layer 2 include:

- latest observed within-quarter level,
- within-quarter change,
- partial-quarter average,
- months-missing count,
- block-level coverage rates,
- factor summaries,
- state uncertainty summaries,
- block-level news,
- absolute news magnitudes,
- release-flow descriptors.

---

## DFM outputs that must be exported

Layer 1 is not complete if it exports only a single nowcast number.

A correct research-grade Layer 1 should export at least:

- DFM nowcasts
- latent states / factors
- news decomposition
- block news summaries
- coverage metrics
- diagnostics
- metadata / manifests describing the export contract

These exports are necessary because Layer 2 is supposed to learn from the **objects produced by the DFM backbone**, not from an opaque black box.

---

## Benchmark logic

The hybrid model should be evaluated against at least the following baselines:

- **AR(p)**:
  - minimal GDP-only benchmark
- **DFM-only**:
  - main benchmark and decisive comparison
- optional simpler mixed-frequency benchmarks:
  - as robustness checks rather than the conceptual core

The DFM-only comparison is the most important one.

If the hybrid model beats AR(p) but not DFM-only, that does **not** establish the main research contribution.

The key scientific claim requires showing improvement over a strong vintage-consistent DFM benchmark.

---

## Evaluation philosophy

Evaluation must be done in **pseudo real time**.

That means:

- for each vintage `v`, estimate or update the model using only data available by `v`,
- generate the nowcast for the relevant target quarter,
- compare against the chosen truth definition only after the fact.

### Main scoring target

The default main truth is:

- **third-release real GDP growth**

Why this choice:
- it is less noisy than first-release GDP,
- but still close to the practical information environment relevant for real-time forecasting.

### Robustness truths

In robustness analysis, compare against:

- latest RTDSM value
- GDPplus

### Main loss metric

A natural primary loss is:

- **RMSFE** over pseudo-real-time forecasts

### Forecast comparison

Because the hybrid model nests the DFM baseline in spirit, forecast comparison should not stop at average error reductions. Use forecast-comparison tools appropriate for nested or realistic out-of-sample settings when the experiment reaches publication-grade evaluation.

---

## Interpretability philosophy

A major strength of this project is that it keeps a **two-layer interpretation system**.

### DFM layer interpretation
The state-space model supports **news decomposition**:

- which releases arrived,
- what the model expected,
- what the surprise was,
- and how those surprises moved the nowcast.

### ML layer interpretation
The residual learner can be interpreted using feature-attribution tools such as SHAP-style analyses.

This gives a two-part narrative:

1. **DFM news:** why the structural nowcast moved
2. **ML correction drivers:** why the hybrid adjusted the structural nowcast further

This interpretability layer is one of the reasons the project is designed as **DFM + ML residual correction** rather than a single black-box model.

---

## What must not be changed casually

Future contributors, assistants, or coding agents should not “simplify” the method by doing any of the following:

1. fitting the DFM on latest revised GDP history instead of vintage-specific GDP history
2. standardizing predictors using the full sample
3. pre-imputing ragged-edge missing values outside the state-space system
4. replacing FRED-MD with FRED-QD as the main Layer 1 predictor flow
5. training ML to forecast GDP directly as the default design
6. training residual models on quarters that do not yet have realized truth
7. storing only headline nowcasts while discarding states/news/coverage/diagnostics
8. converting monthly or quarterly identifiers to month-end or quarter-end timestamps in a way that breaks the repository’s first-of-period semantics

These changes would materially alter the research design.

---

## Recommended workflow

A practical research workflow is:

### Debug phase
Use a smaller, stable, controlled configuration:

- stable subset
- limited factor specification
- small lag-order grid
- short sample window
- focus on EM/Kalman convergence and export correctness

### Research phase
Run the full benchmark design:

- main benchmark window beginning at 2000Q1
- full FRED-MD panel for primary results
- stable subset for robustness
- third-release truth as the main score target
- expanding-window pseudo-real-time design as the main setup
- rolling-window variants only as robustness

The project should move to Layer 2 only after Layer 1 has passed basic diagnostic checks such as:

- stable EM convergence,
- stable factor orientation,
- economically sensible nowcast revisions,
- readable block-level news decomposition,
- and competitive DFM-only performance versus simpler baselines.

---

## Intended repository outputs

A complete version of this repository should eventually make it easy to reproduce:

- the real-time target construction,
- the monthly predictor vintage pipeline,
- the transformed vintage-by-vintage predictor panels,
- the DFM estimation and nowcast generation,
- the news and coverage exports,
- the Layer 2 residual-design table,
- ML model training / tuning / evaluation,
- and publication-style comparisons between AR, DFM, and Hybrid models.

---

## One-sentence summary

This repository implements a **real-time, vintage-aware, mixed-frequency GDP nowcasting system** in which a **state-space Dynamic Factor Model** provides the structural and interpretable backbone, and **machine learning** is used only to **predict and correct the DFM residual** under strict pseudo-real-time information constraints.
