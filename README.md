# UpLift — causal targeting engine

Repository scaffold. **Status: M0 — plan only, no implementation yet.**

Given a completed marketing experiment, decide *who* to treat and *what that is worth* — with
uncertainty stated, not implied.

- Build plan: [PLAN.md](PLAN.md)
- Agent rules: [AGENTS.md](AGENTS.md)
- Data: Hillstrom MineThatData e-mail experiment (public), optional Criteo Uplift.

## Why this exists

Churn-prediction notebooks answer "who will convert". This answers "who converts *because* we
treated them", which is the decision a business actually makes. The deliverable is a decision memo
with a Qini curve, bootstrap intervals and a cost-based targeting threshold — not a model score.

## Target results (to be filled from `make report`, never typed by hand)

| Metric | Baseline | UpLift | Status |
|---|---|---|---|
| Uplift@10% | random targeting (0) | — | not measured |
| Qini coefficient | response-model targeting | — | not measured |
| Incremental margin / ₦ spent | untargeted campaign | — | not measured |

## Not in scope

- Dashboards, SaaS packaging, auth.
- Deep learning. If a boosted tree doesn't beat a two-model baseline, that is the finding.
- Any number that isn't produced by a script in this repo.
