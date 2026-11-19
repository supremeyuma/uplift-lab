"""Figures for the report. Plotting code is excluded from the coverage gate."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from uplift.eval.calibration import CalibrationBin
from uplift.eval.qini import QiniCurve

plt.rcParams["figure.dpi"] = 110


def qini_figure(
    curve: QiniCurve,
    random_line: np.ndarray,
    oracle_line: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.fractions, curve.qini, label="model", color="#1f5f8b")
    ax.plot(curve.fractions, random_line, label="random", linestyle="--", color="#999999")
    ax.plot(curve.fractions, oracle_line, label="oracle", linestyle=":", color="#c0392b")
    ax.set_xlabel("share of population targeted")
    ax.set_ylabel("cumulative incremental conversions")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def uplift_at_k_figure(
    labels: list[str],
    values: list[float],
    lows: list[float],
    highs: list[float],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(values))
    errors = np.array([np.array(values) - np.array(lows), np.array(highs) - np.array(values)])
    ax.bar(positions, values, yerr=errors, capsize=4, color="#1f5f8b")
    ax.set_xticks(positions, labels)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_ylabel("observed uplift in group")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_figure(table: list[CalibrationBin], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    predicted = [row.predicted_mean for row in table]
    observed = [row.observed_uplift for row in table]
    lows = [row.ci_low for row in table]
    highs = [row.ci_high for row in table]
    positions = np.arange(len(table))
    ax.plot(positions, predicted, marker="o", label="predicted", color="#1f5f8b")
    ax.errorbar(
        positions,
        observed,
        yerr=[np.array(observed) - np.array(lows), np.array(highs) - np.array(observed)],
        fmt="s",
        capsize=3,
        label="observed",
        color="#c0392b",
    )
    ax.set_xticks(positions, [str(row.decile + 1) for row in table])
    ax.set_xlabel("score decile (1 = highest)")
    ax.set_ylabel("uplift")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
