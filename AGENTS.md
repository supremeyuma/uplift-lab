# AGENTS.md

Rules for any coding agent working in this repository.

## Read first

`PLAN.md` is the source of truth. It defines milestones, acceptance commands and commit boundaries.
If the plan and the code disagree, the plan wins; if the plan is wrong, propose an edit to it in the
same commit as the fix.

## Workflow

1. Pick the lowest-numbered incomplete milestone. Do not skip ahead.
2. Read the whole milestone section before writing code.
3. Write the tests listed for that milestone first. They should fail for the right reason.
4. Implement until `make ci` passes.
5. Update the checkbox in `PLAN.md`, then commit.

## Commit rules

- Conventional Commits: `feat(scope):`, `fix(scope):`, `test(scope):`, `docs:`, `chore:`.
- One commit per meaningful slice — a loader, a metric, a gate — not one commit per file and not a
  single dump at the end.
- The body states *why*, and any number quoted in the subject must come from a command you ran.
- Never commit raw data, secrets, `.env`, or generated figures larger than 1 MB.

## Engineering rules

- Python 3.11+, type hints everywhere, mypy strict, ruff clean.
- No network access in tests. Use fixtures under `tests/fixtures/` or a seeded generator.
- All randomness takes an explicit seed; tests assert determinism.
- Statistical claims need an interval. A bare point estimate is a bug.
- Do not edit `tests/fixtures/expected_*` to make a test pass. If an expectation is wrong, say why
  in the commit body.
- Update `README.md` numbers only by running `make report` and copying `reports/metrics.json`.

## Definition of done for any task

- Tests added or updated.
- `make ci` green locally.
- Docs touched if behaviour changed.
- No TODO left silently; either fix or add it to `PLAN.md` under the current milestone.
