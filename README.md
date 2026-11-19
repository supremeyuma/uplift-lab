# UpLift — causal targeting engine

Given a completed marketing experiment, decide *who* to treat and *what that is worth* — with
uncertainty stated, not implied.

- Build plan: [PLAN.md](PLAN.md)
- Agent rules: [AGENTS.md](AGENTS.md)
- Case study: [docs/case-study.md](docs/case-study.md)
- Decision memo (regenerated): `reports/decision-memo.md`
- Data: Hillstrom MineThatData e-mail experiment (public), optional Criteo Uplift.

## Why this exists

Churn-prediction notebooks answer "who will convert". This answers "who converts *because* we
treated them", which is the decision a business actually makes. The deliverable is a decision memo
with a Qini curve, bootstrap intervals and a cost-based targeting threshold — not a model score.

## Target results (copied from `reports/metrics.json`, never typed by hand)

Baselines: random targeting for uplift@10, response-model targeting for the Qini coefficient,
untargeted campaign for incremental margin. Evaluation split: 12,891 held-out rows (20% of
Hillstrom), seed 20240501.

| Metric | Baseline | UpLift | 95% interval |
|---|---|---|---|
| Uplift@10% | 0.00% | 1.21% | 95% CI 0.04% – 2.42% |
| Qini coefficient | -0.0496 | 0.0838 | 95% CI -0.0263 – 0.2034 |
| Incremental margin | -₦308,889 | -₦24,862 | 95% CI -₦169,969 – ₦116,933 |

`python scripts/check_readme_numbers.py` fails if any cell above drifts from
`reports/metrics.json`.

## Quick start

```bash
make setup          # uv sync
make data           # fetch Hillstrom once, verify sha256, write splits
make ci             # ruff + mypy strict + pytest with an 85% coverage gate
make report         # reports/metrics.json, figures, decision-memo.md
```

On a fresh clone, `make report` fetches the dataset itself if `data/raw/` is empty.
CI additionally runs the whole pipeline on a committed 2k-row synthetic fixture
(`--fixture`), so the pipeline is exercised without the network.

## What is measured

- **Qini coefficient** — area between the Qini curve and random targeting, normalised by the
  oracle ranking, with a seeded bootstrap 95% interval.
- **Uplift@k** — observed treated-minus-control conversion rate in the top k% by predicted
  uplift, k = 5/10/20%, with bootstrap intervals and response-model/random baselines.
- **Calibration error** — sample-weighted mean absolute gap between predicted and observed uplift
  across score deciles.
- **Incremental margin** — measured incremental conversions in the targeted group × assumed
  margin − treatment cost, with a 95% interval and a full sensitivity grid.

## Not in scope

- Dashboards, SaaS packaging, auth.
- Deep learning. If a boosted tree doesn't beat a two-model baseline, that is the finding.
- Any number that isn't produced by a script in this repo.
