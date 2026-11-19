# ADR-001 — Split strategy

**Status:** accepted · **Date:** 2026-11-20

## Context

The project needs a train/validation/test split that is reproducible on any machine, survives a
re-fetch of the same file, and cannot leak rows between splits. A random split with a stored seed
would satisfy reproducibility locally but would silently change if pandas or numpy's RNG changed,
and it couples the split to row order.

## Decision

Assign each row to a split by hashing its canonical content (all schema columns, fixed formatting)
with `blake2b`, digest size 8, mapped to [0, 1) and cut at 60% / 80%. The split therefore depends
only on the row's values:

- `train` 38,309 rows, `validation` 12,800, `test` 12,891 (64,000 total, seed 20240501).
- The persisted artefact stores validation and test row indices; train is the complement.
- `data/manifests/hillstrom.json` pins the file sha256; a mismatch aborts before any split is used.
- Tests assert the assignment is invariant to row order (property test) and that no row appears in
  two splits.

## Consequences

- Deterministic without depending on library RNG implementations.
- Duplicate rows land in the same split; that is acceptable because they are identical rows.
- The persisted indices depend on the raw file's row order, which the sha256 manifest fixes.
- The test split is used once, after model selection, for the reported metrics. Re-using it for a
  second decision would invalidate the intervals; the case study says so explicitly.
