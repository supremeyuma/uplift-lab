# Case study — who should receive the e-mail?

This is the decision the project exists to make: not "who will convert", but "who converts
*because* we mailed them, and is that worth the spend?" Every number below comes from
`make report` → `reports/metrics.json` on the committed 60/20/20 split (seed 20240501). The
evaluation split is touched once, after model selection on validation, and never tuned afterwards.

## 1. Problem

A retailer has a completed e-mail experiment with three arms: no e-mail, men's e-mail, women's
e-mail. A campaign costs money per customer and only pays when a customer converts because of it.
Two errors matter:

- treating a customer who would have converted anyway (cost with no incremental margin);
- not treating a customer whose conversion is caused by the e-mail (lost margin).

A response model cannot answer this: it ranks high-baseline customers who need no nudge. The
question is a treatment-effect question, so the deliverable is an uplift ranking plus a cost rule,
with an interval on the value.

**What the experiment can support.** A two-sided test of a binary e-mail treatment at the observed
sample size. At 80% power and α=0.05, the minimum detectable effect is **0.1934 percentage points**
on a 0.5726% control conversion rate; the observed 0.4955pp difference has realised power 1.00. The
experiment is therefore adequately powered for the average effect.

## 2. Data

Hillstrom MineThatData e-mail experiment: 64,000 customers, one row each, covariates recency,
history, history segment, mens/womens indicator, zip code, newbie, channel; outcomes visit,
conversion, spend. The raw CSV is fetched once by `make data`, checked against a committed sha256
manifest (`data/manifests/hillstrom.json`), and never committed.

Treatment is collapsed to **any e-mail vs no e-mail** (42,694 vs 21,306). Variables that are
post-treatment (visit, spend given conversion) are never used as features.

Limits that follow from the data: one binary treatment, a short measurement window, US retail in
2008, and no cost or margin field. The economics below are therefore explicit assumptions, not
measurements.

## 3. Method

- **Split.** Rows are assigned by a blake2b hash of their content, so the split does not depend on
  file order: 38,309 train / 12,800 validation / 12,891 test. Test rows are used once, after model
  selection. See [ADR-001](decisions/ADR-001-split-strategy.md).
- **Estimator.** T-learner: separate outcome models for treated and control rows, uplift =
  μ₁(x) − μ₀(x). Training rows only receive out-of-fold predictions from a 5-fold cross-fit; the
  test rows are scored by models fitted on all training rows. Logistic and LightGBM variants share
  the protocol in `models/base.py`; an optional econml `CausalForestDML` path uses the same
  interface.
- **Selection.** The variant with the higher validation Qini coefficient is carried to the test
  split: LightGBM **0.1059** vs logistic **−0.0648**. The test split played no part in this choice.
- **Metrics.** Qini coefficient normalised by the oracle ranking; uplift@5/10/20% as the observed
  treated-minus-control rate difference in the top k%, with 2,000-resample percentile bootstrap
  intervals; calibration by score decile; and the incremental margin of the cost-threshold policy.
  See [ADR-002](decisions/ADR-002-metric-choice.md).
- **Decision rule.** Treat only customers whose predicted uplift exceeds cost/margin, with margin
  ₦5,000 per conversion and treatment cost ₦40 per customer (break-even 0.80%). See
  [ADR-004](decisions/ADR-004-threshold-choice.md).

## 4. Results

Test split, n = 12,891, 2,000 bootstrap resamples.

| Ranking | Qini | Uplift@5% | Uplift@10% | Uplift@20% |
|---|---|---|---|---|
| Random targeting | 0.000 | 0.00% | 0.00% | 0.00% |
| Response model (logistic, p(conversion \| x)) | −0.0496 | 0.44% | 0.51% | 0.30% |
| T-learner LightGBM (selected) | 0.0838 [−0.0263, 0.2034] | 1.01% [−0.71%, 2.75%] | 1.21% [0.04%, 2.42%] | 0.49% [−0.34%, 1.36%] |

Baselines are point estimates; intervals are shown for the selected model. Calibration error
(sample-weighted mean absolute predicted-vs-observed gap across deciles) is **0.0097**.

What the table says, including the parts that are not flattering:

- The uplift model beats the response model on Qini (0.0838 vs −0.0496) and more than doubles the
  observed uplift in the top 10% (1.21% vs 0.51%).
- The Qini interval includes zero. On this test split, "better than random" is not established at
  95% confidence for the whole ranking.
- Only uplift@10% has an interval entirely above zero. Uplift@5% and uplift@20% do not. The
  evidence is concentrated in a narrow band of scores, not a broad gradient.
- The response model is a weak targeting rule at the top of the ranking, and worse than random
  across the whole curve. That is the finding the project was built to expose.

**Economics** (assumed margin ₦5,000, cost ₦40):

| Policy | Targeted | Measured incremental margin | 95% CI |
|---|---|---|---|
| Untargeted campaign | 12,891 | −₦308,889 | — |
| Cost-threshold policy | 4,158 (32.3%) | −₦24,862 | −₦169,969 to ₦116,933 |

The policy improves on the untargeted campaign by about ₦284,000, but its measured value is still
negative at the base assumptions, and its interval includes zero. The model's predicted value
(₦220,986) is not what the experiment delivered in the targeted group: predicted uplift was
over-optimistic for the 32% of customers above the threshold. The honest recommendation in
`reports/decision-memo.md` is **do not run the campaign at these assumptions**.

The sensitivity grid shows where the decision flips. With margin ₦5,000 and cost ₦20, measured
value is +₦47,545; with margin ₦7,500 and cost ₦20 it is +₦138,526. At cost ₦40 every margin in the
grid is negative. Across the grid the measured value ranges from −₦107,802 to +₦138,526. This is
why the memo reports a range and a table rather than a single naira figure.

## 5. Limits

1. **One treatment, one message class.** Hillstrom cannot rank channels or creatives, and the
   treated arms differ in message, not incentive. The project does not establish a creative effect.
2. **Short window.** Conversion and spend are measured shortly after the campaign; repeat purchases
   and long-run effects are invisible, so the value estimate is probably conservative for the
   business and unverifiable from this data.
3. **External validity.** The data are from a 2008 US retailer. The sign of the effect is not
   guaranteed to transfer to another market; the *method* is what transfers.
4. **Assumed economics.** Margin and cost are assumptions, not measurements. The sensitivity grid
   is the honest form of that answer, and its 15 cells are exploratory, not 15 independent tests.
5. **Model risk.** Bootstrap intervals cover sampling variation in the evaluation split. They do
   not cover the possibility that the next campaign behaves differently, nor the instability of
   selecting one learner on a single validation split.
6. **Multiple comparisons.** Three k values, many deciles and a 15-cell grid are reported without
   family-wise correction. Treat individual cells as descriptive.

## 6. Next step

Before any spend:

1. Re-run the cost rule on a fresh holdout with the threshold fixed in advance (the current test
   split has been seen once and cannot arbitrate a second decision).
2. Test the threshold policy against the untargeted campaign in a small randomised pilot, sized
   with `uplift.design.power` for the targeted-group rate difference, not the overall rate.
3. Collect cost and margin data instead of assuming them; the decision is more sensitive to cost
   than to any modelling choice in this project.
4. Only then consider a multi-treatment dataset (Criteo) and the optional causal forest, which is
   behind the same interface and currently unselected.
