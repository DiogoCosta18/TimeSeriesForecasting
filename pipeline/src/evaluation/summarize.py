from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import atomic_write_text


QUESTIONS = [
    "Which decomposition method wins overall?",
    "Does without_stl beat STL methods for any model family?",
    "Does stl_seasonal_naive beat stl_model_all_components?",
    "Does stl_model_all_components help on high evolving seasonality?",
    "Does stl_seasonal_naive help on low spectral entropy?",
    "Which models are most robust to high residual non-normality?",
    "Which models are most robust to high nonlinearity?",
    "Which models are most robust to high spectral entropy?",
    "Which models are most robust to high evolving seasonality?",
    "Which models are most robust to structural breaks?",
    "Which models are most robust to ARCH/volatility clustering?",
    "Does bucket fine-tuning improve over no fine-tuning?",
    "If series-level fine-tuning is enabled, does it improve over bucket fine-tuning?",
    "Does fine-tuning overfit on low-data buckets?",
    "Are results different for M3 vs M4?",
    "Are results different for monthly vs quarterly?",
    "Which model/decomposition/fine-tuning combination should be the default under REL_NAIVE?",
    "Which combinations are too expensive relative to their performance gain?",
]


def _compute_fallback_counts(metrics: pd.DataFrame) -> dict[str, int]:
    if metrics.empty or "model_output_source" not in metrics.columns:
        return {}
    fallback = metrics[~metrics["model_output_source"].str.startswith("trained_", na=False)]
    if fallback.empty:
        return {}
    counts: dict[str, int] = {}
    if "model_family" in fallback.columns:
        for family, g in fallback.groupby("model_family"):
            counts[str(family)] = len(g)
    else:
        counts["total"] = len(fallback)
    return counts


def write_summary(
    path: Path,
    overall: pd.DataFrame,
    bucket: pd.DataFrame,
    metrics: pd.DataFrame,
    notes: list[str],
    *,
    strict_mode: bool = False,
    fallback_counts: dict[str, int] | None = None,
    benchmark_valid: bool | None = None,
) -> None:
    if fallback_counts is None:
        fallback_counts = _compute_fallback_counts(metrics)
    if benchmark_valid is None:
        benchmark_valid = len(fallback_counts) == 0

    lines = ["# Forecasting Experiment Summary", ""]

    # Benchmark validity header
    total_fallbacks = sum(fallback_counts.values())
    if benchmark_valid:
        lines += ["**Benchmark validity: VALID** — all forecasts produced by real trained models.", ""]
    else:
        lines += [
            f"**WARNING: NOT BENCHMARK-VALID** — {total_fallbacks} fallback forecast rows detected.",
            "",
        ]
        if fallback_counts:
            lines += ["Fallback counts by model family:", ""]
            for family, count in sorted(fallback_counts.items()):
                lines.append(f"- `{family}`: {count} rows")
            lines.append("")

    if strict_mode:
        lines += ["Strict mode: ENABLED", ""]
    else:
        lines += ["Strict mode: DISABLED (non-strict run; fallbacks permitted but labeled)", ""]

    if overall.empty:
        lines += ["No completed metrics were available.", ""]
    else:
        best = overall.iloc[0]
        lines += [
            f"Best overall combination by REL_NAIVE is `{best.model_name}` / `{best.decomposition_method}` / `{best.finetuning_mode}` on `{best.frequency}` with mean REL_NAIVE `{best.rel_naive_mean_unclipped:.4f}`.",
            "",
        ]
    for i, q in enumerate(QUESTIONS, 1):
        answer = "Insufficient completed rows for a strong answer."
        if not overall.empty:
            if "decomposition method wins" in q:
                answer = overall.groupby("decomposition_method")["rel_naive_mean_unclipped"].mean().sort_values().index[0]
            elif "bucket fine-tuning" in q:
                ft = overall.groupby("finetuning_mode")["rel_naive_mean_unclipped"].mean().sort_values()
                answer = f"Observed ranking: {', '.join(f'{k}={v:.3f}' for k, v in ft.items())}."
            elif "monthly vs quarterly" in q:
                fr = overall.groupby("frequency")["rel_naive_mean_unclipped"].mean().sort_values()
                answer = f"Frequency means: {', '.join(f'{k}={v:.3f}' for k, v in fr.items())}."
            elif "default under REL_NAIVE" in q:
                row = overall.iloc[0]
                answer = f"`{row.model_family}/{row.model_name}` with `{row.decomposition_method}` and `{row.finetuning_mode}`."
            elif "too expensive" in q:
                answer = "Inspect leaderboard cost columns; combinations with high training_time_seconds and no REL_NAIVE gain are candidates."
            else:
                answer = "See the corresponding feature and bucket leaderboards for the ranked result; smoke runs are directional only."
        lines += [f"## {i}. {q}", "", str(answer), ""]
    if notes:
        lines += ["## Environment and Fallback Notes", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    atomic_write_text(path, "\n".join(lines))
