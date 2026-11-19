# PLAN — UpLift (causal targeting engine)

Status: **M2 complete** · Owner: supreme · Target completion: 6 milestones / ~2 weeks

This document is the contract for the coding agent building this repo. Work one milestone at a
time. A milestone is done only when its **acceptance commands** pass from a clean clone. Update the
checkboxes in the same commit as the work.

---

## 1. What the repo must prove

1. You can design and audit an experiment (power, MDE, what the test cannot support).
2. You can estimate heterogeneous treatment effects and evaluate them with uplift metrics — not
   accuracy.
3. You can convert a model output into a business decision with a cost matrix and an uncertainty
   statement.

An interviewer should be able to run `make report` and get the same numbers as the README.

## 2. Non-goals

- No dashboards, no web app, no auth.
- No deep learning, no AutoML. Baselines must be explicit and beatable.
- No metric in the README typed by hand — every one is emitted by `scripts/report.py`.
- No test that touches the network.

## 3. Repo layout (target)

```
uplift-lab/
  pyproject.toml            # uv + ruff + mypy strict + pytest
  Makefile                  # setup | data | test | lint | report | ci
  data/
    raw/                    # gitignored, fetched by `make data`
    manifests/              # sha256 + shape + provenance for each dataset
  src/uplift/
    data/loader.py          # download, cache, validate, split by row hash
    data/schema.py          # column contract + treatment/outcome definitions
    design/power.py         # MDE, required n, realised power
    models/response.py      # p(y|x) baseline
    models/tlearner.py      # two-model uplift
    models/causal_forest.py # optional econml path
    eval/qini.py            # Qini curve + coefficient
    eval/uplift_at_k.py     # uplift@k with bootstrap CI
    eval/calibration.py     # predicted vs observed uplift by decile
    decision/cost.py        # cost matrix, targeting threshold
    decision/memo.py        # renders decision memo (md)
    report.py               # figures + metrics json
  tests/
    unit/                   # fast, deterministic
    properties/             # hypothesis-based invariants
    integration/            # end-to-end on a fixture sample
    fixtures/               # small CSVs + a synthetic DGP generator
  docs/
    case-study.md           # filled at M6, mirrors the EvalGate case study
    decisions/              # ADR-001.. per milestone decision
```

## 4. Agent operating rules

- **Gate discipline:** do not start milestone N+1 until `make ci` is green for N.
- **TDD for invariants:** the milestone's core statistical property gets a test *before* the code.
- **Determinism:** every random seed is explicit and configurable; tests assert repeatability.
- **Commits:** Conventional Commits. One commit per meaningful slice, never per file. Suggested
  units are listed per milestone; inline them if a slice is small.
- **Honesty:** if a metric is worse than baseline, report it and explain it in the case study. Do
  not tune on test data.
- **No fabricated results:** `make report` writes `reports/metrics.json`; README numbers are copied
  from that file in the same commit.

## 5. Milestones

### M0 — Scaffold (0.5 day)

**Build**
- `pyproject.toml` (uv, ruff, mypy strict, pytest), `Makefile`, `src/uplift/__init__.py`.
- GitHub Actions: lint → typecheck → test on push and PR.
- `Makefile` targets: `setup`, `lint`, `test`, `ci`.

**Acceptance**
```bash
make setup && make ci     # passes on empty test suite (at least one smoke test)
```

**Commit**
- `chore: scaffold package, tooling and CI`

---

### M1 — Data contract and experiment audit (1 day)

**Build**
- `data/loader.py`: download Hillstrom (MineThatData) once, cache under `data/raw/`, write
  `data/manifests/hillstrom.json` with sha256, row count, column dtypes, fetch timestamp.
- `data/schema.py`: frozen dataclass describing treatment column (`segment`), outcome
  (`conversion`, `spend`), covariates, and a `validate()` that raises on schema drift.
- Deterministic split by stable row hash → `train / validation / test` (60/20/20), persisted.
- `design/power.py`: compute MDE at 80% power, α=0.05, given observed n and baseline rate, plus the
  realised power for the observed effect.

**Tests**
- `test_manifest_hash_matches_file`
- `test_validate_rejects_missing_treatment_column`
- `test_split_is_deterministic_and_disjoint` (property test over shuffled input order)
- `test_no_row_leaks_across_splits`
- `test_power_mde_shrinks_with_sample_size` (property test)
- `test_realised_power_matches_reference_for_known_case`

**Acceptance**
```bash
make data && make test
python -m uplift.design.power --data hillstrom   # prints MDE + realised power
```
Manifest and split files exist and are committed (small JSON only, not raw data).

**Commit**
- `feat(data): add Hillstrom loader, schema contract and deterministic splits`
- `feat(design): add power/MDE analysis with tests`

---

### M2 — Baselines and uplift estimators (2 days)

**Build**
- `models/response.py`: p(conversion|x) baseline (logistic + LightGBM), used for both the response
  baseline and the T-learner.
- `models/tlearner.py`: separate treated/control models, uplift = μ₁(x) − μ₀(x), with cross-fitting
  (K-fold, out-of-fold predictions only).
- `models/causal_forest.py`: optional econml `CausalForestDML`, same interface, same split.
- `models/base.py`: `UpliftModel` protocol — `fit(X, t, y)`, `predict_uplift(X) -> np.ndarray`.

**Tests**
- `test_tlearner_recovers_constant_effect_within_tolerance` (synthetic DGP, ATE known)
- `test_tlearner_recovers_known_heterogeneity_by_segment` (effect only in a known subgroup)
- `test_predictions_are_out_of_fold` (assert no train row predicted by a model that saw it)
- `test_all_models_share_interface` (parametrised over implementations)

**Acceptance**
```bash
make test
python -m uplift.models.tlearner --check-recovery   # prints recovered vs true ATE on fixture
```

**Commit**
- `feat(models): add response baseline, T-learner and causal forest behind one protocol`
- `test(models): synthetic recovery tests for ATE and subgroup heterogeneity`

---

### M3 — Evaluation: Qini, uplift@k, calibration (2 days)

**Build**
- `eval/qini.py`: Qini curve + coefficient, computed against random and response-model targeting.
- `eval/uplift_at_k.py`: uplift@5/10/20% with bootstrap CI (≥1000 resamples, seeded).
- `eval/calibration.py`: predicted vs observed uplift by decile, with CI, plus a calibration error
  summary.
- `eval/report.py`: one JSON + one figure per metric, written to `reports/`.

**Tests**
- `test_qini_of_random_is_zero_within_ci` (property)
- `test_qini_of_perfect_ranking_equals_analytic_max` (constructed case)
- `test_uplift_at_k_bootstrap_ci_covers_truth` (synthetic DGP, repeated draws ≥95%)
- `test_calibration_flags_overconfident_scores` (injected miscalibration is detected)
- `test_metrics_are_deterministic_with_seed`

**Acceptance**
```bash
make test && python -m uplift.report --split test
# reports/metrics.json contains qini, uplift@10, ci_low, ci_high
```

**Commit**
- `feat(eval): add Qini, uplift@k with bootstrap CI, and calibration`
- `feat(report): emit metrics.json and figures from a single command`

---

### M4 — Decision layer (1 day)

**Build**
- `decision/cost.py`: configurable cost matrix (treatment cost, margin per conversion), expected
  incremental value per customer, targeting threshold, and the optimal top-k policy.
- Sensitivity: value as a function of assumed margin and treatment cost (grid, saved to CSV).
- `decision/memo.py`: renders `reports/decision-memo.md` with the recommendation, the excluded
  negative-uplift segment, the uncertainty statement and the decision the test cannot support.

**Tests**
- `test_threshold_is_monotone_in_margin`
- `test_negative_uplift_segment_is_excluded`
- `test_expected_value_matches_hand_computed_example` (golden numbers)
- `test_memo_renders_without_missing_fields` (snapshot-ish, assert required headings)

**Acceptance**
```bash
make report
# reports/decision-memo.md exists, states naira value, CI and excluded segment
```

**Commit**
- `feat(decision): add cost-based targeting threshold and decision memo`

---

### M5 — Reproducible report and CI gate (1 day)

**Build**
- `make report` runs end-to-end from a clean clone (data fetch cached, splits regenerated).
- CI job that runs the pipeline on a 2k-row fixture and asserts metrics.json schema + ranges.
- `scripts/check_readme_numbers.py`: fails if README table disagrees with `reports/metrics.json`.

**Tests**
- `test_report_runs_from_fixture_under_2_minutes`
- `test_readme_numbers_gate_catches_drift` (mutate a number, expect failure)

**Acceptance**
```bash
make ci && python scripts/check_readme_numbers.py
```

**Commit**
- `feat(ci): end-to-end report job and README/metrics consistency gate`

---

### M6 — Documentation, case study, release (1 day)

**Build**
- Fill README target-results table from `reports/metrics.json`.
- `docs/case-study.md`: problem → data → method → results table vs baselines → limits → next step.
- `docs/decisions/ADR-00*.md` for: split strategy, metric choice, model choice, threshold choice.
- Tag `v1.0.0`. Then (only then) add the project block to the CV.

**Acceptance**
```bash
git clone <repo> /tmp/uplift && cd /tmp/uplift && make setup && make ci && make report
# README numbers match reports/metrics.json; case study contains no unsupported claims
```

**Commit**
- `docs: case study, ADRs and measured README`
- `chore(release): v1.0.0`

---

## 6. Metrics specification (fixed now, so they can't be redefined later)

| Metric | Definition | Reported with |
|---|---|---|
| Qini coefficient | area between Qini curve and random line, normalised | bootstrap 95% CI |
| Uplift@k | mean predicted-vs-observed uplift in top k% (k=5,10,20) | CI + baseline |
| Incremental margin | Σ over targeted of (p_conv_treat − p_conv_control) × margin − cost | range from sensitivity grid |
| Calibration error | mean abs difference, predicted vs observed uplift by decile | per-decile table |

Baselines that must always appear: random targeting, response-model targeting, untargeted campaign.

## 7. Test strategy summary

- **Unit:** schema, splitting, power maths, metric edge cases.
- **Property (hypothesis):** determinism, CI coverage, monotonicity, leakage.
- **Synthetic DGP:** the only way to prove an uplift model is correct — known ATE and known
  heterogeneity must be recovered within stated tolerance.
- **Integration:** fixture-sized end-to-end run under 2 minutes, no network.
- **Gate:** README ↔ metrics.json consistency.

Coverage target: 85% on `src/uplift` (excluding plotting), enforced in CI.

## 8. Risks and decisions to record

| Risk | Mitigation |
|---|---|
| Hillstrom has one binary treatment, limited covariates | State limits in the memo; optional Criteo path for scale |
| Uplift metrics are noisy | Bootstrap CIs everywhere; never report a point estimate alone |
| Overfitting the test split | Test split touched once, at M6, recorded in ADR |
| Economics are assumptions | Sensitivity grid, not a single naira figure |

## 9. Done checklist

- [x] M0 scaffold + CI
- [x] M1 data contract, splits, power analysis
- [x] M2 baselines + T-learner + optional causal forest
- [ ] M3 Qini, uplift@k, calibration
- [ ] M4 decision layer + memo
- [ ] M5 reproducible report + consistency gate
- [ ] M6 case study, ADRs, v1.0.0, CV block added
