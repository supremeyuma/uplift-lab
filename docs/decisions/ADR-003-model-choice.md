# ADR-003 — Model choice

**Status:** accepted · **Date:** 2026-11-20

## Context

The estimator must produce per-customer uplift, be auditable, and run on a laptop without a GPU.
Candidate families: a response model used as a targeting score (baseline), a two-model T-learner,
and a causal forest.

## Decision

- **Protocol.** `models/base.UpliftModel`: `fit(X, t, y)` and `predict_uplift(X) -> ndarray`.
  All evaluation and decision code depends only on this protocol.
- **Baselines.** Pooled logistic response model `p(conversion | x)` (with a LightGBM variant
  available), reported as the response-targeting baseline. It is deliberately strong: if it ranked
  as well as the uplift model, the incremental claim would not be supported.
- **T-learner.** Separate arm models with 5-fold cross-fitting. Training rows receive only
  out-of-fold predictions; test rows are scored by models fitted on all training rows. A fold whose
  arm has a single outcome class falls back to a constant probability rather than crashing.
- **Selection.** Logistic vs LightGBM T-learner is chosen on **validation Qini only**: LightGBM
  0.1059 vs logistic −0.0648. The test split is evaluated once, after this choice.
- **Causal forest.** econml `CausalForestDML` is available behind the same protocol via the
  `causal-forest` extra. It is optional, is not installed in the default environment, and was not
  selected. The wrapper is tested offline with an injected backend.

## Consequences

- The selected LightGBM T-learner is the model behind every metric in the README.
- Cross-fitting makes the calibration and threshold diagnostics honest, at the cost of fitting
  2 × K + 2 models per run.
- Selecting on a single validation split is itself a source of variance; the case study lists this
  under limits. A repeated-validation selection would be the next methodological improvement.
- The causal forest is unselected and unevidenced on this dataset. It is infrastructure, not a
  result.
