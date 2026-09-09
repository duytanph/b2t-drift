"""CLI: build the drift curve and error-breakdown figures from one or more
`scripts/03_evaluate.py` `per_session_metrics.csv` outputs.

Thin wrapper only -- reads the CSV(s) `03_evaluate.py` already computed
(PER, bootstrap CI, sub/del/ins breakdown, median seq_len), combines them,
and plots. No PER, CI, or alignment computation happens here; this script
only aggregates and visualizes numbers `b2t_drift.evaluation` already
produced (`CLAUDE.md`: no logic belongs in `scripts/`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_MIN_TRIALS = 50
DEFAULT_OUT_DIR = Path("figures")

_INT_FIELDS = (
    "days_since_cut", "n_trials", "substitutions", "deletions", "insertions",
    "n_reference_phonemes", "silence_deletions", "phoneme_deletions",
)
_FLOAT_FIELDS = (
    "per", "per_ci_lower", "per_ci_upper", "substitution_rate", "deletion_rate",
    "insertion_rate", "median_seq_len", "decoded_length_ratio", "eval_seconds",
)


def _load_metrics_csv(path: Path) -> list[dict]:
    """Read one `per_session_metrics.csv`, casting numeric columns and
    sorting by `days_since_cut` (the drift curve's x-axis)."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in _INT_FIELDS:
            row[field] = int(row[field])
        for field in _FLOAT_FIELDS:
            row[field] = float(row[field])
        row["passes_threshold"] = row["passes_threshold"] in ("True", "true", "1")
    rows.sort(key=lambda r: r["days_since_cut"])
    return rows


def _sibling_summary(metrics_csv_path: Path) -> dict | None:
    """Read the `evaluation_summary.json` `03_evaluate.py` writes alongside
    `per_session_metrics.csv` in the same directory, or None if absent."""
    summary_path = metrics_csv_path.parent / "evaluation_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as f:
        return json.load(f)


def _linear_fit_summary(rows: list[dict]) -> tuple[float, float]:
    """Slope (PER per day) and intercept of an ordinary least-squares fit
    of PER against days_since_cut. Purely descriptive -- see main()'s print
    statement for the caveat against over-interpreting it."""
    days = np.array([r["days_since_cut"] for r in rows], dtype=np.float64)
    per = np.array([r["per"] for r in rows], dtype=np.float64)
    slope, intercept = np.polyfit(days, per, 1)
    return float(slope), float(intercept)


def _print_text_summary(label: str, rows: list[dict]) -> None:
    first, last = rows[0], rows[-1]
    abs_change = last["per"] - first["per"]
    rel_change = abs_change / first["per"] if first["per"] else float("nan")
    slope, _ = _linear_fit_summary(rows)
    print(f"\n[04_drift_curve] === {label} ===")
    print(f"  earliest held-out day: +{first['days_since_cut']}d ({first['session_id']}), PER={first['per']:.4f}")
    print(f"  latest held-out day:   +{last['days_since_cut']}d ({last['session_id']}), PER={last['per']:.4f}")
    print(f"  absolute change: {abs_change:+.4f}  relative change: {rel_change:+.2%}")
    print(f"  descriptive linear fit (NOT a model -- {len(rows)} points, no significance claimed): "
          f"slope={slope * 100:+.4f} PER per 100 days")


def _plot_drift_curve(sources: list[dict], out_path: Path, min_trials: int) -> None:
    fig, (ax_per, ax_len) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1],
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, source in enumerate(sources):
        color = colors[i % len(colors)]
        rows = source["rows"]
        days = [r["days_since_cut"] for r in rows]
        per = [r["per"] for r in rows]
        lower_err = [r["per"] - r["per_ci_lower"] for r in rows]
        upper_err = [r["per_ci_upper"] - r["per"] for r in rows]

        ax_per.errorbar(
            days, per, yerr=[lower_err, upper_err], color=color, marker="o",
            markersize=5, linewidth=1.5, capsize=3, label=source["label"],
        )

        low_trial = [r for r in rows if r["n_trials"] < min_trials]
        if low_trial:
            ax_per.scatter(
                [r["days_since_cut"] for r in low_trial], [r["per"] for r in low_trial],
                marker="x", s=70, color=color, zorder=5,
                label=f"{source['label']} (n_trials < {min_trials})",
            )

        if source["training_val_per"] is not None:
            ax_per.axhline(
                source["training_val_per"], color=color, linestyle="--", linewidth=1, alpha=0.7,
            )

        ax_len.plot(
            days, [r["median_seq_len"] for r in rows], color=color, marker="o", markersize=4,
        )

    ax_per.set_ylabel("PER (fraction)")
    ax_per.set_title("Drift curve: PER vs. days since training cut")
    ax_per.legend(fontsize=8, loc="best")
    ax_per.grid(alpha=0.3)

    ax_len.set_xlabel("Days since cut")
    ax_len.set_ylabel("Median seq_len")
    ax_len.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[04_drift_curve] wrote {out_path}")


def _plot_error_breakdown(sources: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    panels = [
        ("substitution_rate", "Substitution rate"),
        ("deletion_rate", "Deletion rate"),
        ("insertion_rate", "Insertion rate"),
    ]

    for i, source in enumerate(sources):
        color = colors[i % len(colors)]
        rows = source["rows"]
        days = [r["days_since_cut"] for r in rows]
        for ax, (field, _) in zip(axes, panels):
            ax.plot(days, [r[field] for r in rows], color=color, marker="o", markersize=4, label=source["label"])

    for ax, (_, title) in zip(axes, panels):
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_title("Error breakdown vs. days since training cut")
    axes[-1].set_xlabel("Days since cut")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[04_drift_curve] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, nargs="+", required=True,
        help="One or more per_session_metrics.csv paths (scripts/03_evaluate.py output).",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help=(
            "Display label per --metrics input, same order and count. Default: each input's "
            "split_name from its sibling evaluation_summary.json, or its parent directory name "
            "if that file is missing."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=(
            "Output directory for drift_curve.png, error_breakdown.png, and drift_curve.csv "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-trials", type=int, default=DEFAULT_MIN_TRIALS,
        help="Sessions with n_trials below this are marked with a distinct marker, not dropped (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.metrics):
        parser.error(f"--labels given {len(args.labels)} value(s) but --metrics has {len(args.metrics)}.")

    sources = []
    for i, path in enumerate(args.metrics):
        rows = _load_metrics_csv(path)
        summary = _sibling_summary(path)
        if args.labels is not None:
            label = args.labels[i]
        elif summary is not None:
            label = summary.get("split_name", path.parent.name)
        else:
            label = path.parent.name
        training_val_per = summary.get("checkpoint_val_per") if summary else None
        sources.append({"label": label, "rows": rows, "path": path, "training_val_per": training_val_per})

    args.out.mkdir(parents=True, exist_ok=True)

    _plot_drift_curve(sources, args.out / "drift_curve.png", args.min_trials)
    _plot_error_breakdown(sources, args.out / "error_breakdown.png")

    combined_path = args.out / "drift_curve.csv"
    all_fieldnames = list(_load_metrics_csv(args.metrics[0])[0].keys()) if sources else []
    fieldnames = ["source", *all_fieldnames]
    with combined_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source in sources:
            for row in source["rows"]:
                writer.writerow({"source": source["label"], **row})
    print(f"[04_drift_curve] wrote {combined_path}")

    for source in sources:
        _print_text_summary(source["label"], source["rows"])


if __name__ == "__main__":
    main()
