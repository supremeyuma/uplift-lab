"""End-to-end pipeline: train, select on validation, evaluate on the test split.

Everything reported is produced here and written to ``reports/metrics.json``.
README numbers are copied from that file; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uplift.config import (
    BOOTSTRAP_RESAMPLES,
    ROOT,
    SEED,
    SPLIT_FRACTIONS,
    ensure_dir,
    reports_dir,
)
from uplift.data.loader import ensure_dataset, load_frame, load_split_indices, split_frame
from uplift.data.schema import HILLSTROM, DatasetSchema
from uplift.data.synthetic import make_experiment
from uplift.decision.cost import (
    CostMatrix,
    find_excluded_segment,
    segment_uplift,
    sensitivity_grid,
    summarize_policy,
    value_range,
)
from uplift.decision.memo import write_memo
from uplift.eval.calibration import calibration_error, calibration_table
from uplift.eval.figures import calibration_figure, qini_figure, uplift_at_k_figure
from uplift.eval.qini import QiniResult, qini_coefficient, qini_curve, qini_with_ci
from uplift.eval.report import write_json
from uplift.eval.uplift_at_k import UpliftAtK, uplift_at_k_with_ci
from uplift.models.features import covariates
from uplift.models.response import ResponseModel, logistic_learner
from uplift.models.tlearner import TLearner, make_tlearner

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "hillstrom_fixture.csv"
FIXTURE_ROWS = 2000
UPLIFT_KS: tuple[float, ...] = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    source: str


def load_fixture(path: Path = FIXTURE_PATH) -> pd.DataFrame:
    if not path.exists():
        frame = make_experiment(
            n=FIXTURE_ROWS, seed=SEED, logit_effect=0.5, heterogeneity=0.6
        ).frame
        ensure_dir(path.parent)
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    HILLSTROM.validate(frame)
    return frame


def load_splits(*, fixture: bool = False) -> SplitFrames:
    if fixture:
        frame = load_fixture()
        parts = split_frame(frame)
        return SplitFrames(
            train=parts["train"],
            validation=parts["validation"],
            test=parts["test"],
            source="fixture:synthetic-2k",
        )
    ensure_dataset()
    frame = load_frame()
    indices = load_split_indices()
    return SplitFrames(
        train=frame.iloc[indices["train"]].reset_index(drop=True),
        validation=frame.iloc[indices["validation"]].reset_index(drop=True),
        test=frame.iloc[indices["test"]].reset_index(drop=True),
        source="hillstrom:64k",
    )


def _xy(frame: pd.DataFrame, schema: DatasetSchema) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = covariates(frame, schema)
    t = schema.is_treated(frame).to_numpy(dtype=int)
    y = frame["conversion"].to_numpy(dtype=int)
    return X, t, y


def _selection(
    frames: SplitFrames, schema: DatasetSchema, seed: int
) -> tuple[TLearner, dict[str, Any], ResponseModel]:
    """Pick the T-learner by validation Qini; the test split plays no part."""
    X_fit, t_fit, y_fit = _xy(frames.train, schema)
    X_val, _, y_val = _xy(frames.validation, schema)
    t_val = schema.is_treated(frames.validation).to_numpy(dtype=int)

    candidates: dict[str, TLearner] = {
        "tlearner-logistic": make_tlearner("logistic", seed=seed),
        "tlearner-lightgbm": make_tlearner("lightgbm", seed=seed, n_folds=3),
    }
    scores: dict[str, float] = {}
    for name, candidate in candidates.items():
        candidate.fit(X_fit, t_fit, y_fit)
        scores[name] = qini_coefficient(y_val, t_val, candidate.predict_uplift(X_val))

    best_name = max(scores, key=lambda name: scores[name])
    chosen = candidates[best_name]

    response = ResponseModel(logistic_learner(seed), schema=schema, seed=seed)
    response.fit(X_fit, t_fit, y_fit)
    selection = {
        "selected": best_name,
        "criterion": "validation Qini coefficient (point estimate)",
        "validation_qini": scores,
    }
    return chosen, selection, response


def build_report(
    *,
    split: str = "test",
    fixture: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    output_dir: Path | None = None,
    schema: DatasetSchema = HILLSTROM,
) -> dict[str, Any]:
    if split not in SPLIT_FRACTIONS:
        raise ValueError(f"unknown split {split!r}")
    frames = load_splits(fixture=fixture)
    evaluation = getattr(frames, split)

    model, selection, response = _selection(frames, schema, seed)

    X_eval, t_eval, y_eval = _xy(evaluation, schema)
    uplift_score = model.predict_uplift(X_eval)
    response_score = response.predict_proba(X_eval)

    qini_ci: QiniResult = qini_with_ci(
        y_eval, t_eval, uplift_score, resamples=resamples, seed=seed
    )
    qini_payload: dict[str, Any] = {
        "value": qini_ci.value,
        "ci_low": qini_ci.ci_low,
        "ci_high": qini_ci.ci_high,
        "resamples": qini_ci.resamples,
        "baselines": {
            "random": 0.0,
            "response_model": qini_coefficient(y_eval, t_eval, response_score),
        },
    }

    uplift_payload: dict[str, Any] = {}
    uplift_results: dict[str, UpliftAtK] = {}
    for k in UPLIFT_KS:
        label = f"uplift_at_{round(k * 100)}"
        result = uplift_at_k_with_ci(
            y_eval, t_eval, uplift_score, k, resamples=resamples, seed=seed
        )
        uplift_results[label] = result
        uplift_payload[label] = {
            "value": result.uplift,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "k": result.k,
            "n_targeted": result.n_targeted,
            "treated_share": result.treated_share,
            "baselines": {
                "random": 0.0,
                "response_model": uplift_at_k_with_ci(
                    y_eval, t_eval, response_score, k, resamples=200, seed=seed
                ).uplift,
            },
        }

    table = calibration_table(
        y_eval, t_eval, uplift_score, bins=10, resamples=200, seed=seed
    )
    calibration_payload = {
        "error": calibration_error(y_eval, t_eval, uplift_score),
        "deciles": [
            {
                "decile": row.decile,
                "n": row.n,
                "predicted_mean": row.predicted_mean,
                "observed_uplift": row.observed_uplift,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
            }
            for row in table
        ],
    }

    target_dir = ensure_dir(output_dir if output_dir is not None else reports_dir())
    rate_difference = float(y_eval[t_eval == 1].mean() - y_eval[t_eval == 0].mean())
    curve = qini_curve(y_eval, t_eval, uplift_score)
    perfect_score = y_eval * (2.0 * t_eval - 1.0)
    oracle_curve = qini_curve(y_eval, t_eval, perfect_score)
    qini_figure(
        curve,
        curve.random_line(rate_difference),
        oracle_curve.qini,
        target_dir / "qini.png",
        f"Qini — {selection['selected']} ({split})",
    )

    labels = [f"top {int(k * 100)}% (uplift)" for k in UPLIFT_KS]
    uplifts = [uplift_results[key] for key in uplift_payload]
    uplift_at_k_figure(
        labels,
        [u.uplift for u in uplifts],
        [u.ci_low for u in uplifts],
        [u.ci_high for u in uplifts],
        target_dir / "uplift-at-k.png",
        f"Uplift@k with bootstrap 95% CI ({split})",
    )
    calibration_figure(
        table, target_dir / "calibration.png", f"Calibration by score decile ({split})"
    )

    cost = CostMatrix()
    policy = summarize_policy(
        y_eval, t_eval, uplift_score, cost, resamples=resamples, seed=seed
    )
    grid = sensitivity_grid(y_eval, t_eval, uplift_score)
    grid_low, grid_high = value_range(grid)
    segment_rows = segment_uplift(
        y_eval,
        t_eval,
        uplift_score,
        evaluation["history_segment"].to_numpy(),
        resamples=300,
        seed=seed,
    )
    if not segment_rows:  # pragma: no cover - guarded by decile/segment counts
        raise RuntimeError("no segment had both arms and enough rows")
    excluded = find_excluded_segment(segment_rows)
    lowest = min(segment_rows, key=lambda row: row["observed_uplift"])

    untargeted_incremental = rate_difference * len(evaluation)
    untargeted_value = (
        untargeted_incremental * cost.margin_per_conversion
        - cost.treatment_cost * len(evaluation)
    )

    decision_payload: dict[str, Any] = {
        "assumptions": policy.assumptions,
        "threshold": policy.threshold,
        "n_targeted": policy.n_targeted,
        "share_targeted": policy.share_targeted,
        "predicted_value": policy.predicted_value,
        "measured_incremental_conversions": policy.measured_incremental_conversions,
        "measured_value": policy.measured_value,
        "ci_low": policy.ci_low,
        "ci_high": policy.ci_high,
        "roi": policy.roi,
        "resamples": resamples,
        "range": {"low": grid_low, "high": grid_high},
        "sensitivity": grid,
        "untargeted_campaign": {
            "n": len(evaluation),
            "incremental_conversions": untargeted_incremental,
            "value": untargeted_value,
        },
    }

    payload: dict[str, Any] = {
        "dataset": schema.name,
        "source": frames.source,
        "split": split,
        "seed": seed,
        "bootstrap_resamples": resamples,
        "n": len(evaluation),
        "model": selection,
        "metrics": {
            "qini": qini_payload,
            **uplift_payload,
            "calibration_error": calibration_payload["error"],
            "incremental_margin": {
                "value": policy.measured_value,
                "ci_low": policy.ci_low,
                "ci_high": policy.ci_high,
                "predicted_value": policy.predicted_value,
                "n_targeted": policy.n_targeted,
                "range": {"low": grid_low, "high": grid_high},
                "currency": cost.currency,
                "baselines": {"untargeted_campaign": untargeted_value},
            },
        },
        "calibration": calibration_payload,
        "decision": decision_payload,
        "segments": {
            "by": "history_segment",
            "table": segment_rows,
            "excluded": excluded,
            "lowest": lowest,
        },
        "baselines": {
            "random_targeting": "uplift 0 by construction; untargeted campaign in the decision memo",
            "response_model": "p(conversion | x) logistic, trained on the same rows",
        },
    }
    write_json(target_dir / "metrics.json", payload)
    write_memo(payload, target_dir / "decision-memo.md")
    return payload


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the UpLift measurement pipeline")
    parser.add_argument("--split", default="test", choices=sorted(SPLIT_FRACTIONS))
    parser.add_argument("--fixture", action="store_true", help="2k synthetic fixture, no network")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = build_report(
        split=args.split,
        fixture=args.fixture,
        resamples=args.resamples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    metrics = payload["metrics"]
    print(f"dataset          : {payload['dataset']} ({payload['source']})")
    print(f"evaluation split : {payload['split']}  n={payload['n']:,}")
    print(f"selected model   : {payload['model']['selected']}")
    for name, value in payload["model"]["validation_qini"].items():
        print(f"  validation qini: {name:<20} {_fmt(float(value))}")
    print(
        f"qini             : {_fmt(metrics['qini']['value'])} "
        f"[{_fmt(metrics['qini']['ci_low'])}, {_fmt(metrics['qini']['ci_high'])}]"
    )
    for key in ("uplift_at_5", "uplift_at_10", "uplift_at_20"):
        block = metrics[key]
        print(
            f"{key:<17}: {_fmt(block['value'])} "
            f"[{_fmt(block['ci_low'])}, {_fmt(block['ci_high'])}]"
        )
    print(f"calibration error: {_fmt(metrics['calibration_error'])}")
    decision = payload["decision"]
    print(
        f"targeted         : {decision['n_targeted']:,} "
        f"({decision['share_targeted']:.1%}), break-even {decision['threshold']:.2%}"
    )
    print(
        f"incremental margin: {decision['assumptions']['currency_code']} "
        f"{decision['measured_value']:,.0f} "
        f"[{decision['ci_low']:,.0f}, {decision['ci_high']:,.0f}]"
    )
    print(f"written          : {(args.output_dir or reports_dir()) / 'metrics.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
