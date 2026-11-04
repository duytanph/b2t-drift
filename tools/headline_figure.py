"""figures/headline.png -- the one figure that carries the whole project.

TOP: drift curve (PER vs. days since calibration) at all four cut points
(s05, s10, s15, s20). s05 is drawn emphasized -- full saturation, filled
markers, bootstrap 95% CI, the training-block validation PER (0.158) as a
reference line, and hollow diamonds for its below-50-trial sessions. s10,
s15 and s20 are drawn as lighter context -- thin lines, no markers, no CI,
no below-threshold distinction -- so the panel reads as one emphasized
curve plus three supporting ones, not four competing series.

BOTTOM: Arm 3b recovery fraction vs. days, s05 only, one line per N, all 7
sessions (including the +184d/+259d gap-fill), 0% reference line.

Shared linear x-axis (days since calibration) so the two panels read as
one story: the decoder degrades at every cut point tried (top), and
recalibration's power to fix that degradation collapses right along with
it, on s05 specifically (bottom).

No new numbers: every value plotted already exists in
results/drift_curve.csv (all four `source` values) and
results/recalibration_recovery_fraction.csv. No checkpoint, no HDF5 array
load -- reads committed results CSVs only, so this is reproducible from
what's already in the repo.

Run with: uv run tools/headline_figure.py
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
CONTEXT_BLUE = "#aac8ea"
N_STYLE = {10: ("#86b6ef", "o"), 25: ("#3987e5", "s"), 50: ("#1c5cab", "^"), 100: ("#0d366b", "D")}
VAL_PER = 0.15765
EMPHASIZED_CUT = "s05"
CONTEXT_CUTS = ("s10", "s15", "s20")

DEFAULT_DRIFT_CSV = Path("results/drift_curve.csv")
DEFAULT_RECOVERY_CSV = Path("results/recalibration_recovery_fraction.csv")
DEFAULT_OUTPUT = Path("figures/headline.png")
DEFAULT_DPI = 160

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def _load_drift_rows(path: Path) -> dict[str, list[dict]]:
    by_cut: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(path)):
        by_cut.setdefault(row["source"], []).append(row)
    for rows in by_cut.values():
        rows.sort(key=lambda r: int(r["days_since_cut"]))
    return by_cut


def _load_recovery_rows(path: Path) -> dict[int, list[dict]]:
    by_n: dict[int, list[dict]] = {}
    for row in csv.DictReader(open(path)):
        by_n.setdefault(int(row["n_calibration"]), []).append(row)
    for rows in by_n.values():
        rows.sort(key=lambda r: int(r["days_since_cut"]))
    return by_n


def _draw_top(ax, drift_by_cut: dict[str, list[dict]]) -> None:
    # Context cuts first (lighter, drawn underneath; no CI, no threshold marker).
    for cut in CONTEXT_CUTS:
        rows = drift_by_cut[cut]
        x = [int(r["days_since_cut"]) for r in rows]
        y = [float(r["per"]) for r in rows]
        ax.plot(x, y, color=CONTEXT_BLUE, linewidth=1.1, alpha=0.9, zorder=1)

    # Emphasized cut last (on top): full style, CI, below-threshold diamonds.
    drift_rows = drift_by_cut[EMPHASIZED_CUT]
    above = [r for r in drift_rows if r["passes_threshold"] == "True"]
    below = [r for r in drift_rows if r["passes_threshold"] == "False"]

    x_all = [int(r["days_since_cut"]) for r in drift_rows]
    y_all = [float(r["per"]) for r in drift_rows]
    ax.plot(x_all, y_all, color=BLUE, linewidth=1.6, zorder=2)

    xa = [int(r["days_since_cut"]) for r in above]
    ya = [float(r["per"]) for r in above]
    ea_lo = [float(r["per"]) - float(r["per_ci_lower"]) for r in above]
    ea_hi = [float(r["per_ci_upper"]) - float(r["per"]) for r in above]
    ax.errorbar(xa, ya, yerr=[ea_lo, ea_hi], fmt="o", color=BLUE, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1, elinewidth=1.1, capsize=2.5, zorder=4)

    xb = [int(r["days_since_cut"]) for r in below]
    yb = [float(r["per"]) for r in below]
    eb_lo = [float(r["per"]) - float(r["per_ci_lower"]) for r in below]
    eb_hi = [float(r["per_ci_upper"]) - float(r["per"]) for r in below]
    ax.errorbar(xb, yb, yerr=[eb_lo, eb_hi], fmt="D", color=SURFACE, markeredgecolor=ORANGE,
                markersize=7, markeredgewidth=1.8, elinewidth=1.1, capsize=2.5, ecolor=ORANGE, zorder=5)

    ax.axhline(VAL_PER, color=INK2, linewidth=1.5, linestyle="--", zorder=1)

    legend = [
        Line2D([0], [0], color=BLUE, linewidth=1.6, marker="o", markersize=8,
               markeredgecolor=SURFACE, label="s05 (emphasized): held-out session (>=50 trials)"),
        Line2D([0], [0], marker="D", color=SURFACE, markeredgecolor=ORANGE, markersize=8,
               markeredgewidth=1.8, linewidth=0, label="s05 below 50-trial threshold (excluded from fits)"),
        Line2D([0], [0], color=CONTEXT_BLUE, linewidth=1.6,
               label="s10, s15, s20 (context: no CI, no threshold marker)"),
        Line2D([0], [0], color=INK2, linewidth=1.5, linestyle="--",
               label=f"s05 training-block validation PER ({VAL_PER:.3f})"),
    ]
    # Headroom above the 0.84 data ceiling (ylim extends to 1.0) is reserved
    # for this legend so it never overlaps a curve, at any cut, anywhere on
    # the x-axis -- the alternative (placing it inside the data region) has
    # no x-position where some cut's curve isn't already passing through.
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=10.5, bbox_to_anchor=(0.002, 1.0),
              ncol=2, columnspacing=1.2, handlelength=1.8, handletextpad=0.6, labelspacing=0.5)

    ax.set_ylabel("phoneme error rate (PER)", fontsize=14)
    ax.set_title("The decoder degrades steeply, then plateaus - at all four cut points",
                 loc="left", fontsize=17, fontweight="bold", pad=10)
    ax.set_ylim(0.05, 1.0)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def _draw_bottom(ax, rec_by_n: dict[int, list[dict]]) -> None:
    ax.axhline(0, color=INK2, linewidth=1.5, zorder=2)
    legend = []
    for n in (10, 25, 50, 100):
        color, marker = N_STYLE[n]
        rows = rec_by_n[n]
        x = [int(r["days_since_cut"]) for r in rows]
        y = [100 * float(r["recovery_fraction_mean"]) for r in rows]
        e = [100 * float(r["recovery_fraction_sd_over_splits"]) for r in rows]
        small = [int(r["n_eval_dedup_min"]) < 30 for r in rows]
        ax.plot(x, y, color=color, linewidth=1.8, zorder=3)
        for xi, yi, ei, sm in zip(x, y, e, small):
            ax.errorbar([xi], [yi], yerr=[ei], fmt=marker, color=color, markersize=9,
                        markerfacecolor=SURFACE if sm else color, markeredgecolor=color if sm else SURFACE,
                        markeredgewidth=1.8 if sm else 1.2, elinewidth=1.2, capsize=3, zorder=4)
        minutes = st.mean(float(r["calibration_minutes_mean"]) for r in rows)
        legend.append(Line2D([0], [0], marker=marker, color=color, markersize=9, markeredgecolor=SURFACE,
                              linewidth=1.8, label=f"N={n}  (~{minutes:.0f} min of speech)"))

    n10 = rec_by_n[10]
    x9, y9 = int(n10[0]["days_since_cut"]), 100 * float(n10[0]["recovery_fraction_mean"])
    ax.annotate("10 labeled sentences here made\nthe decoder WORSE than doing nothing",
                (x9, y9), xytext=(95, 30), textcoords="offset points", fontsize=10.5, color=RED,
                ha="left", va="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    for x0, x1 in [(58, 184), (259, 329)]:
        ax.axvspan(x0, x1, color="#fde8e2", zorder=0)
    ax.text(121, 47, "step\ndrop", ha="center", va="top", fontsize=10, color="#c0552f")
    ax.text(294, 47, "step\ndrop", ha="center", va="top", fontsize=10, color="#c0552f")

    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=11.5, bbox_to_anchor=(0.99, 0.99))
    ax.set_ylabel("share of the drift gap\nrecovered by full fine-tuning (%)", fontsize=13.5)
    ax.set_xlabel("days between training and the evaluation session", fontsize=14.5, labelpad=10)
    ax.set_title("Recalibration's power to fix it collapses right along with it (s05)",
                 loc="left", fontsize=17, fontweight="bold", pad=10)
    ax.set_ylim(-18, 52)
    ax.set_xlim(-15, 620)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drift-csv", type=Path, default=DEFAULT_DRIFT_CSV,
        help="Drift-curve results CSV with one row per (cut, session) and a "
        "'source' column naming the cut (default: %(default)s).",
    )
    parser.add_argument(
        "--recovery-csv", type=Path, default=DEFAULT_RECOVERY_CSV,
        help="Arm 3b recovery-fraction results CSV, s05 only (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output image path (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help="Output resolution (default: %(default)s).",
    )
    args = parser.parse_args()

    drift_by_cut = _load_drift_rows(args.drift_csv)
    missing = [c for c in (EMPHASIZED_CUT, *CONTEXT_CUTS) if c not in drift_by_cut]
    if missing:
        raise SystemExit(f"{args.drift_csv} is missing cut(s): {missing}")
    rec_by_n = _load_recovery_rows(args.recovery_csv)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 11), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.2},
    )
    for ax in (ax_top, ax_bot):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=12)

    _draw_top(ax_top, drift_by_cut)
    _draw_bottom(ax_bot, rec_by_n)

    fig.suptitle("An intracortical speech decoder, calibrated once and never touched again, over 20 months",
                 x=0.01, y=0.998, ha="left", va="top", fontsize=13.5, color=INK2)
    fig.text(0.01, 0.975, "Top: all four training cut points (s05, s10, s15, s20). "
              "Bottom: s05 only.", ha="left", va="top", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.93, bottom=0.06)

    args.output.parent.mkdir(exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
