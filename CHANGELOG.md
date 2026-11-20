# Changelog

## v1.0.0 — 2026-11-20

First reproducible release.

- Hillstrom loader with a committed sha256 manifest and content-hash 60/20/20 splits.
- Power/MDE analysis: MDE 0.1934pp at 80% power, realised power 1.00 for the observed 0.4955pp.
- Response baseline, cross-fitted T-learner (logistic and LightGBM) and an optional econml causal
  forest behind one `UpliftModel` protocol.
- Qini coefficient, uplift@5/10/20% and calibration with seeded 2,000-resample bootstrap intervals.
- Cost-based targeting threshold, measured-value interval, sensitivity grid and decision memo.
- `make report` regenerates `reports/metrics.json`, the figures and the memo; README numbers are
  machine-checked against it.
- `make ci`: ruff, mypy strict, 120 tests, 85% coverage gate; CI also runs the pipeline on a
  committed 2k synthetic fixture and regenerates the real report behind the README gate.

Known limits (see `docs/case-study.md`): the Qini interval includes zero, and the cost rule's
measured value is negative at the base assumed margin and cost, so the memo recommends not running
the campaign at those assumptions.
