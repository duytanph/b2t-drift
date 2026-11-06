"""figures/recalibration_arms.png + results/recalibration_summary.csv.

Design (dataviz skill): one line per arm, fixed categorical slot per arm
(never re-assigned), line style + marker shape as a second identity channel
(solid = needs labels, dashed = features only), thin 1.8pt lines, >=8px
markers, hairline grid, direct-comparable y scales (every panel spans the same
PER range, centred on that session's Arm 0), Arm 0 as a neutral reference line.
The summary CSV is the table view for the colours that fall under 3:1 contrast
on the light surface.

INPUT: results/recalibration_grid.csv (committed), via the sibling module
tools/task3_analysis_tables.py (five-arm recalibration grid: Phase 3's main
comparison run -- `docs/FINDINGS.md`: "Full fine-tuning is the best use of a
fixed participant-time budget..." and the entries around it). No checkpoint,
no HDF5 load -- reproducible from what's already in the repo.

Run with: uv run tools/task3_make_figure.py
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from task3_analysis_tables import NS, SESSIONS, TOTAL_TRIALS, cell, days, have, mean, sd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ARM_STYLE = {
    # arm: (categorical slot colour, marker, linestyle, legend label)
    "3b": ("#2a78d6", "o", "-", "Arm 3b  full fine-tune (15.0M params, lr 1e-3)"),
    "3a": ("#eb6834", "s", "-", "Arm 3a  output layer only (21k params, lr 1e-4)"),
    "2": ("#1baf7a", "^", "-", "Arm 2  day layer, backbone frozen (1k params, lr 1e-5)"),
    "1b": ("#eda100", "D", "--", "Arm 1b  Procrustes alignment (no labels)"),
    "1a": ("#e87ba4", "v", "--", "Arm 1a  CORAL alignment (no labels)"),
}
DRAW_ORDER = ["1a", "1b", "2", "3a", "3b"]  # headline arm on top
Y_SPAN = 0.28
Y_BELOW, Y_ABOVE = 0.23, 0.05

DEFAULT_FIGURE_OUTPUT = Path("figures/recalibration_arms.png")
DEFAULT_TABLE_OUTPUT = Path("results/recalibration_summary.csv")
DEFAULT_DPI = 170

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE_OUTPUT, help="Output image path (default: %(default)s).")
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT, help="Output table CSV path (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output resolution (default: %(default)s).")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.6))
    flat = axes.ravel()

    for ax, s in zip(flat[:5], SESSIONS):
        arm0_full = mean("0", s, 0, "arm0_dedup")
        ax.set_xscale("log")
        ax.set_xlim(2.0, 36.0)
        ax.set_ylim(arm0_full - Y_BELOW, arm0_full + Y_ABOVE)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax.axhline(arm0_full, color=INK2, linewidth=1.5, zorder=2)

        for arm in DRAW_ORDER:
            color, marker, ls, _ = ARM_STYLE[arm]
            xs, ys, es = [], [], []
            for n in NS:
                if have(arm, s, n):
                    xs.append(mean(arm, s, n, "min"))
                    ys.append(mean(arm, s, n, "per_dedup"))
                    es.append(sd(arm, s, n, "per_dedup"))
            ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, linestyle=ls,
                        linewidth=1.8, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                        elinewidth=1.1, capsize=2.5, zorder=3 + DRAW_ORDER.index(arm))

        ax.set_xticks([2.5, 5, 10, 20, 30])
        ax.set_xticklabels(["2.5", "5", "10", "20", "30"])
        ax.minorticks_off()
        ax.set_xlabel("minutes of new-day calibration data", fontsize=10)
        ax.set_ylabel("held-out PER", fontsize=10)
        ax.set_title(f"+{days[s]} days after training   ({TOTAL_TRIALS[s]} labeled trials)",
                     loc="left", fontsize=11.5, color=INK, pad=26)

        top = ax.secondary_xaxis("top")
        top.set_xscale("log")
        top.set_xticks([mean("0", s, n, "min") for n in NS if have("0", s, n)])
        top.set_xticklabels([f"N={n}" for n in NS if have("0", s, n)], fontsize=8.5, color=MUTED)
        top.minorticks_off()
        top.tick_params(length=2)
        top.spines["top"].set_color(AXIS)

    # Legend / notes panel
    lax = flat[5]
    lax.axis("off")
    handles = [
        Line2D([0], [0], color=ARM_STYLE[a][0], marker=ARM_STYLE[a][1], linestyle=ARM_STYLE[a][2],
               linewidth=1.8, markersize=7, markeredgecolor=SURFACE, label=ARM_STYLE[a][3])
        for a in ["3b", "3a", "2", "1b", "1a"]
    ]
    handles.append(Line2D([0], [0], color=INK2, linewidth=1.5, label="Arm 0  no adaptation (day-0 decoder)"))
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5,
               labelcolor=INK, handlelength=2.8, borderaxespad=0, labelspacing=0.9)
    # Derived from the grid, not hardcoded, so this can't drift out of sync
    # with the panel titles' own TOTAL_TRIALS-derived labels above.
    missing_100 = [s for s in SESSIONS if not have("0", s, 100)]
    n100_line = (
        "N=100 is impossible at "
        + " and ".join(f"+{days[s]} d ({TOTAL_TRIALS[s]} trials)" for s in missing_100)
        + "; those points are absent, not zero."
    )

    NOTES = "\n".join([
        "Solid lines need transcribed sentences; dashed lines use",
        "recorded neural features only (transcripts discarded).",
        "",
        "Error bars: ±1 sd over 5 random calibration subsets. Every",
        f"panel's y-axis spans the same {Y_SPAN:.2f} of PER, so vertical gaps",
        "compare across panels.",
        "",
        *textwrap.wrap(n100_line, width=57),
        "",
        "Arm 2's rate is capped by its failure at higher rates on +329 d.",
        "",
        "Sentence-de-duplicated PER (Arm 2 not refit on 32 leak-affected cells;",
        "shown at its original PER there).",
    ])
    lax.text(0.0, 0.50, NOTES, transform=lax.transAxes, fontsize=9.2, color=INK2, va="top", linespacing=1.3)

    fig.suptitle("Full fine-tuning recovers PER at every elapsed time; label-free alignment and small labeled updates barely move it  (sentence-de-duplicated PER)",
                 x=0.012, y=0.985, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.952, "Held-out phoneme error rate vs. minutes of new-day calibration, by evaluation session "
                           "(decoder trained on sessions 1-5)", fontsize=10.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.075, hspace=0.42, wspace=0.22)

    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_output, dpi=args.dpi)
    print(f"wrote {args.figure_output}")

    # ---- table view ----
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.table_output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "session_id", "days_since_cut", "n_calibration", "calibration_minutes_mean",
                    "per_mean", "per_sd_over_splits", "arm0_per_mean", "improvement_mean",
                    "improvement_sd_over_splits", "splits_worse_than_arm0", "n_splits"])
        for arm in ["1a", "1b", "2", "3a", "3b"]:
            for s in SESSIONS:
                for n in NS:
                    if not have(arm, s, n):
                        continue
                    d = cell[(arm, s, n)]
                    w.writerow([
                        arm, s, days[s], n, round(st.mean(d["min"]), 3), round(st.mean(d["per"]), 5),
                        round(st.stdev(d["per"]), 5), round(st.mean(d["arm0"]), 5), round(st.mean(d["impr"]), 5),
                        round(st.stdev(d["impr"]), 5), sum(p > a for p, a in zip(d["per"], d["arm0"])), len(d["per"]),
                    ])
    print(f"wrote {args.table_output}")


if __name__ == "__main__":
    main()
