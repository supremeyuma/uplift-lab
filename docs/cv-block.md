# CV project block

> **UpLift — causal targeting engine** (Python, uv, LightGBM, scikit-learn, Hypothesis)
> Built an end-to-end uplift-modelling pipeline on the 64k-row Hillstrom e-mail experiment:
> content-hash 60/20/20 splits, cross-fitted T-learner with logistic and LightGBM arms behind one
> protocol, Qini/uplift@k/calibration metrics with 2,000-resample bootstrap intervals, and a
> cost-based targeting memo. Model selection used validation Qini only; the test split was evaluated
> once (uplift@10% 1.21pp, 95% CI 0.04–2.42pp; Qini 0.0838, CI −0.026 to 0.203). The cost rule's
> measured value was −₦24,862 (CI −₦169,969 to ₦116,933), so the memo recommends not running the
> campaign at the assumed margin and cost. 120 tests, mypy strict, 85% coverage gate, deterministic
> report and README↔metrics consistency gate in CI.

The numbers in this block are the values in `reports/metrics.json` on the current commit; the
README is the canonical, machine-checked copy.
