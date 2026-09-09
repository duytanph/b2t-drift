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
# Present only when the input CSV came from `03_evaluate.py --shuffled-control`
# (Task 2's substitution_rate_among_emitted/n_correct) or
# `--shuffled-control` (Task 3's effective_signal trio) -- cast if present,
# left as strings (harmlessly unused) otherwise. effective_signal_ci_* can
# individually be the empty string (no CI for that session -- donor
# rejection broke pairing, see shuffled_control.build_shuffled_batches).
_OPTIONAL_FLOAT_FIELDS = (
    "substitution_rate_among_emitted", "per_shuffled", "effective_signal",
    "effective_signal_ci_lower", "effective_signal_ci_upper",
)
_OPTIONAL_INT_FIELDS = ("n_correct", "n_shuffle_rejected_no_donor")


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
        for field in _OPTIONAL_FLOAT_FIELDS:
            if field in row and row[field] != "":
                row[field] = float(row[field])
        for field in _OPTIONAL_INT_FIELDS:
            if field in row and row[field] != "":
                row[field] = int(row[field])
        row["passes_threshold"] = row["passes_threshold"] in ("True", "true", "1")
    rows.sort(key=lambda r: r["days_since_cut"])
    return rows


def _has_effective_signal(rows: list[dict]) -> bool:
    return bool(rows) and "effective_signal" in rows[0] and rows[0]["effective_signal"] != ""


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


def _print_effective_signal_summary(label: str, rows: list[dict]) -> None:
    sig_rows = [r for r in rows if r.get("effective_signal", "") != ""]
    if not sig_rows:
        return
    print(f"\n[04_drift_curve] === {label}: effective_signal (per_shuffled - per_real) ===")
    for r in sig_rows:
        print(f"  +{r['days_since_cut']}d ({r['session_id']}): effective_signal={r['effective_signal']:+.4f}")

    days = np.array([r["days_since_cut"] for r in sig_rows], dtype=np.float64)
    signal = np.array([r["effective_signal"] for r in sig_rows], dtype=np.float64)
    slope, _ = np.polyfit(days, signal, 1)
    print(f"  descriptive linear fit: slope={slope * 100:+.4f} effective_signal per 100 days")

    below_threshold = [(r["days_since_cut"], r["effective_signal"]) for r in sig_rows if r["effective_signal"] < 0.05]
    if below_threshold:
        first_day, first_val = min(below_threshold, key=lambda t: t[0])
        print(f"  first falls below 0.05 at +{first_day}d (effective_signal={first_val:.4f})")
    else:
        print("  never falls below 0.05 in the observed range")

    reaches_zero = any(r["effective_signal"] <= 0.0 for r in sig_rows)
    print(f"  reaches zero (<=0) within the observed range: {reaches_zero}")


def _plot_drift_curve(sources: list[dict], out_path: Path, min_trials: int) -> None:
    any_effective_signal = any(_has_effective_signal(s["rows"]) for s in sources)
    n_panels = 3 if any_effective_signal else 2
    height_ratios = [3, 2, 1] if any_effective_signal else [3, 1]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 9 if any_effective_signal else 7), sharex=True,
        height_ratios=height_ratios,
    )
    ax_per = axes[0]
    ax_signal = axes[1] if any_effective_signal else None
    ax_len = axes[-1]
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

        if ax_signal is not None:
            if _has_effective_signal(rows):
                sig_days, sig, sig_lo, sig_hi = [], [], [], []
                for r in rows:
                    if r["effective_signal"] == "":
                        continue
                    sig_days.append(r["days_since_cut"])
                    sig.append(r["effective_signal"])
                    # CI may be individually missing (donor-rejection pairing
                    # break for that session) -- fall back to no error bar.
                    has_ci = r["effective_signal_ci_lower"] != "" and r["effective_signal_ci_upper"] != ""
                    sig_lo.append(r["effective_signal"] - r["effective_signal_ci_lower"] if has_ci else 0.0)
                    sig_hi.append(r["effective_signal_ci_upper"] - r["effective_signal"] if has_ci else 0.0)
                ax_signal.errorbar(
                    sig_days, sig, yerr=[sig_lo, sig_hi], color=color, marker="o",
                    markersize=5, linewidth=1.5, capsize=3,
                )
            else:
                print(f"[04_drift_curve] note: {source['label']} has no effective_signal column "
                      f"(not evaluated with --shuffled-control) -- omitted from that panel.")

        ax_len.plot(
            days, [r["median_seq_len"] for r in rows], color=color, marker="o", markersize=4,
        )

    ax_per.set_ylabel("PER (fraction)")
    ax_per.set_title("Drift curve: PER vs. days since training cut")
    ax_per.legend(fontsize=8, loc="best")
    ax_per.grid(alpha=0.3)

    if ax_signal is not None:
        ax_signal.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax_signal.axhline(0.05, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax_signal.set_ylabel("effective_signal\n(per_shuffled - per_real)")
        ax_signal.grid(alpha=0.3)

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
        _print_effective_signal_summary(source["label"], source["rows"])


if __name__ == "__main__":
    main()
