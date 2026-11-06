"""figures/recalibration_recovery_fraction.png (+ results/recalibration_recovery_fraction.csv
as its table view).

Arm 3b only, all 7 evaluated sessions (+9/+42/+58/+184/+259/+329/+597d -- +184d
and +259d fill what was originally a 271-day gap between +58d and +329d, leaving
two smaller unmeasured intervals: 126 days (+58d to +184d) and 70 days (+259d to
+329d)). Recovery fraction = (Arm0 PER - Arm3b PER) / (Arm0 PER - REFERENCE_PER),
sentence-de-duplicated PER. One line per N (ordinal -> one-hue blue ramp), error
bars = sd over 5 calibration subsets, direct labels at line ends.

CORRECTED FRAMING (docs/FINDINGS.md: "Recovery fraction's 'two sharp steps'
withdrawn: three measured levels, two transitions of unresolved sharpness",
2026-11-05). This figure previously titled itself "Recovery falls in two steps,
not a smooth decline" and shaded the two apparent drops. Both were withdrawn: a
straight line between the +58d and +329d values predicts the +184d/+259d
readings about as well as the step framing did, so the two transitions could be
abrupt or gradual -- this data cannot tell. What IS supported: three measured
levels (early/mid/late), and two intervals with no evaluated session in them.
The shaded bands below now mark those two UNMEASURED intervals, derived from the
session days actually present in the input (not hardcoded), with an assertion
that each band's edges are two measured days with nothing evaluated between
them. The title reports ranges, not an average, for the early group specifically
because collapsing +9/+42/+58d into one mean was itself hiding a real rise
within that group.

INPUT: results/recalibration_grid.csv (committed), produced by the Arm 3b
recalibration grid run (`docs/FINDINGS.md`: "Recalibration slows the decline but
does not restore function", 2026-09-28, extended by "Task 1's gap-fill...",
2026-09-28). This script reads that file directly -- no checkpoint, no HDF5
array load -- so it is reproducible from what is already in the repo.

Run with: uv run tools/recovery_figure.py
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REFERENCE_PER = 0.15765  # s05 validation PER measured WITHIN its training sessions (optimistic ceiling)
NS = [10, 25, 50, 100]
# Below this many evaluation trials (after de-duplication), a cell is too small to
# trust (docs/FINDINGS.md: "...29 eval trials, too small to trust"). Only +58d/N=50
# (29) falls below it; every other cell in the grid is comfortably above.
MIN_RELIABLE_EVAL_TRIALS = 30
# The two intervals the retracted "two sharp steps" claim was about, identified by the
# sessions that bound them. Their day values are looked up from the input below, not
# hardcoded, and asserted to be adjacent in the sorted list of measured elapsed times.
UNMEASURED_GAP_SESSIONS = [
    ("t15.2023.10.22", "t15.2024.02.25"),  # +58d -> +184d
    ("t15.2024.05.10", "t15.2024.07.19"),  # +259d -> +329d
]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND_COLOR = "#e4e3dc"
STYLE = {10: ("#86b6ef", "o"), 25: ("#3987e5", "s"), 50: ("#1c5cab", "^"), 100: ("#0d366b", "D")}

DEFAULT_INPUT = Path("results/recalibration_grid.csv")
DEFAULT_FIGURE_OUTPUT = Path("figures/recalibration_recovery_fraction.png")
DEFAULT_TABLE_OUTPUT = Path("results/recalibration_recovery_fraction.csv")
DEFAULT_DPI = 170

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def _load_arm3b_cells(path: Path):
    """Read the Arm 3b rows and return (sessions-in-day-order, days, per-cell stats)."""
    rows = [r for r in csv.DictReader(open(path)) if r["arm"] == "3b"]
    days = {r["session_id"]: int(r["days_since_cut"]) for r in rows}
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["session_id"], int(r["n_calibration"]))
        a0, per = float(r["arm0_per_dedup"]), float(r["per_dedup"])
        gap = a0 - REFERENCE_PER
        cell[k]["frac"].append((a0 - per) / gap)
        cell[k]["min"].append(float(r["calibration_minutes"]))
        cell[k]["nev"].append(int(r["n_eval_dedup"]))
        cell[k]["a0"].append(a0)
        cell[k]["per"].append(per)
        cell[k]["gap"].append(gap)
        cell[k]["rec"].append(a0 - per)
    sessions = sorted(days, key=lambda sid: days[sid])
    return sessions, days, cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Recalibration grid CSV with per-split Arm 3b/Arm 0 PER rows (default: %(default)s).",
    )
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE_OUTPUT, help="Output image path (default: %(default)s).")
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT, help="Output table CSV path (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output resolution (default: %(default)s).")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found. This script needs the committed Arm 3b recalibration grid "
            f"(one row per session/N/calibration-split; see this script's docstring for the FINDINGS "
            f"entries that produced it). Pass --input to point at a different copy."
        )

    def m(k, f): return st.mean(cell[k][f])
    def s(k, f): return st.stdev(cell[k][f])
    def have(k): return k in cell

    sessions, days, cell = _load_arm3b_cells(args.input)

    def reliable(sid, n):
        return min(cell[(sid, n)]["nev"]) >= MIN_RELIABLE_EVAL_TRIALS

    # The title's ranges must not be set by a flagged (too-few-eval-trial) cell:
    # docs/FINDINGS.md "Recovery fraction's 'two sharp steps' withdrawn...", addendum.
    early = [100 * m((sid, 50), "frac") for sid in sessions if days[sid] <= 58 and reliable(sid, 50)]
    mid = [100 * m((sid, 50), "frac") for sid in sessions if 184 <= days[sid] <= 259]
    late = [100 * m((sid, 50), "frac") for sid in sessions if days[sid] >= 329]

    # Derive the unmeasured-interval band edges from the data, not from literals: look
    # up each bounding session's actual elapsed-time value, and assert nothing measured
    # falls between the two -- i.e. they are adjacent in the sorted list of days present.
    sorted_days = sorted(set(days.values()))
    bands = []
    for sid0, sid1 in UNMEASURED_GAP_SESSIONS:
        x0, x1 = days[sid0], days[sid1]
        i0 = sorted_days.index(x0)
        assert sorted_days[i0 + 1] == x1, (
            f"{sid0} (+{x0}d) and {sid1} (+{x1}d) are not adjacent measured elapsed "
            f"times -- a session was evaluated between them that this band would hide."
        )
        bands.append((x0, x1))

    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(-15, 780)
    ax.set_ylim(-20, 58)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=INK2, linewidth=1.3, zorder=2)

    for n in NS:
        color, marker = STYLE[n]
        sess = sorted((sid for sid in sessions if have((sid, n))), key=lambda sid: days[sid])
        if not sess:
            continue
        x = [days[sid] for sid in sess]
        y = [100 * m((sid, n), "frac") for sid in sess]
        e = [100 * s((sid, n), "frac") for sid in sess]
        small = [not reliable(sid, n) for sid in sess]
        for xi, yi, ei, sm in zip(x, y, e, small):
            ax.errorbar([xi], [yi], yerr=[ei], color=color, marker=marker, linewidth=0, markersize=8,
                        markerfacecolor=SURFACE if sm else color, markeredgecolor=color if sm else SURFACE,
                        markeredgewidth=1.8 if sm else 1.2, elinewidth=1.2, capsize=3, zorder=4)
        ax.plot(x, y, color=color, linewidth=1.8, zorder=3)
        minutes = st.mean(m((sid, n), "min") for sid in sess)
        ax.annotate(f"N={n}  (~{minutes:.0f} min)", (x[-1], y[-1]), xytext=(9, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color=INK)

    # Shade the two intervals with NO evaluated session, not the apparent drops across them.
    for x0, x1 in bands:
        ax.axvspan(x0, x1, color=BAND_COLOR, zorder=0)
    ax.legend(handles=[Patch(facecolor=BAND_COLOR, edgecolor=AXIS, linewidth=0.8,
                              label="interval with no measured session")],
              loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9.5)

    ax.set_xlabel("days between training and the evaluation session", fontsize=10.5, labelpad=8)
    ax.set_ylabel("share of the drift gap recovered  (%)", fontsize=10.5)
    ax.set_xticks([9, 42, 58, 184, 259, 329, 597])
    ax.set_xticklabels(["9", "42", "58", "184", "259", "329", "597"])
    fig.suptitle(
        f"Recovery fraction vs elapsed time (N=50): ~{min(early):.0f}-{max(early):.0f}% through +58d,\n"
        f"~{st.mean(mid):.0f}% at +184-259d, ~{min(late):.0f}-{max(late):.0f}% from +329d on "
        f"(shaded = no session measured)",
        x=0.012, y=0.985, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK,
    )
    fig.text(0.012, 0.895, "Arm 3b (full fine-tune), sentence-de-duplicated PER. Recovery = (Arm 0 PER - Arm 3b PER) / (Arm 0 PER - 0.158). "
                           "Error bars: ±1 sd over 5 calibration subsets.", fontsize=9.5, color=INK2, ha="left")
    fig.text(0.012, 0.012, "0% = no better than no adaptation; 100% = back to 0.158, the s05 model's validation PER measured inside its training sessions "
                          "(an optimistic ceiling, so these are lower bounds on recoverability).\n"
                          f"Hollow marker = fewer than {MIN_RELIABLE_EVAL_TRIALS} evaluation trials left after de-duplication -- the only such cell in this grid is +58d, N=50 (29); its value is plotted "
                          "but excluded from the title's +58d-group range above.\nN=10/100 not run at +184d/+259d (gap-fill covered N=25/50 only); N=100 impossible at +58d/+597d (too few trials).\n"
                          "Shaded intervals contain no evaluated session (126d, 70d): the data cannot distinguish an abrupt change from a gradual one across either one. "
                          "Below 0% at +9d, N=10: fine-tuning on 10 sentences hurt.",
           fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.25)

    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_output, dpi=args.dpi)
    print(f"wrote {args.figure_output}")

    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.table_output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "days_since_cut", "n_calibration", "calibration_minutes_mean", "arm0_per_dedup_mean",
                    "arm3b_per_dedup_mean", "drift_gap_mean", "recovered_mean", "recovery_fraction_mean",
                    "recovery_fraction_sd_over_splits", "n_eval_dedup_min", "reference_per"])
        for sid in sessions:
            for n in NS:
                if have((sid, n)):
                    k = (sid, n)
                    w.writerow([sid, days[sid], n, round(m(k, "min"), 3), round(m(k, "a0"), 5), round(m(k, "per"), 5),
                                round(m(k, "gap"), 5), round(m(k, "rec"), 5), round(m(k, "frac"), 5),
                                round(s(k, "frac"), 5), min(cell[k]["nev"]), REFERENCE_PER])
    print(f"wrote {args.table_output}")


if __name__ == "__main__":
    main()
