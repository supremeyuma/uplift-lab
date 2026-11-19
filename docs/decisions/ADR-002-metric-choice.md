# ADR-002 — Metric choice

**Status:** accepted · **Date:** 2026-11-20

## Context

Accuracy, AUC and log-loss measure how well a model predicts outcomes, not whether it ranks
customers by the *causal* effect of treating them. A model can have excellent AUC and no uplift
signal: it ranks customers who convert anyway.

## Decision

Report four things, each with uncertainty:

1. **Qini coefficient.** Area between the cumulative incremental-conversion curve and the random
   line, normalised by the analytic oracle area, with a 2,000-resample percentile bootstrap 95%
   interval. 0 = random, 1 = oracle.
2. **Uplift@k** for k = 5%, 10%, 20%. Observed treated-minus-control rate difference inside the top
   k% by score, with a bootstrap interval and the response-model/random baselines.
3. **Calibration error.** Sample-weighted mean absolute gap between predicted and observed uplift
   across score deciles, with per-decile intervals.
4. **Incremental margin.** Measured incremental conversions in the targeted group × assumed margin
   − treatment cost, with a bootstrap interval and a sensitivity grid.

All intervals come from `eval/bootstrap.py`, which is seeded, so repeated runs of `make report`
produce identical numbers (asserted in tests).

## Consequences

- Qini and uplift@k answer different questions (whole-ranking quality vs the top of the ranking).
  Reporting both prevents a good top-5% result from hiding a flat curve.
- The Qini coefficient is normalised by the realised outcomes of the evaluation sample, so its
  interval includes the sampling noise of that sample. The observed interval on Hillstrom includes
  zero; the case study reports that rather than suppressing it.
- Calibration is reported even when it is unflattering; it is what makes the negative measured
  margin in the memo interpretable.
- No accuracy metric is reported for the uplift models. Accuracy is not the decision.
