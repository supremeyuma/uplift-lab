# ADR-004 — Threshold choice and economics

**Status:** accepted · **Date:** 2026-11-20

## Context

A ranking is not a decision. The business needs to know which customers to treat and what that is
worth. Hillstrom contains no cost or margin fields, so any monetary answer is an assumption.

## Decision

Treat a customer when expected incremental value is positive:

    uplift(x) × margin − treatment_cost > 0
    threshold = treatment_cost / margin

Base assumptions: **margin ₦5,000 per conversion**, **treatment cost ₦40 per customer**, giving a
break-even uplift of **0.80 percentage points**. Both are explicit constants in
`decision/cost.py`, not tuned on test data.

The memo reports:

- the measured value of the policy (observed uplift in the targeted group × margin − cost),
  with a 2,000-resample bootstrap interval;
- the model's predicted value, clearly separated from the measured value;
- the untargeted-campaign baseline;
- a 5 × 3 sensitivity grid over margin (₦2,500–₦7,500) and cost (₦20–₦80);
- the range across the grid.

## Consequences

At base assumptions the policy targets 4,158 of 12,891 test customers (32.3%). The measured value
is **−₦24,862** with a 95% interval of −₦169,969 to ₦116,933, against **−₦308,889** for the
untargeted campaign. The recommendation in the memo is therefore **do not run the campaign at these
assumptions**: the measured value is negative and the interval includes zero, even though the
policy beats the untargeted baseline.

The grid flips the decision in three cells (margin ₦5,000–₦7,500 with cost ₦20). The memo and
case study present this as a range, not a single confident number. The 15 cells are descriptive,
not independently significant tests.
