"""Standalone diagnostic: does sentence-corpus composition shift across the
20-month recording timeline, and what does each session actually contain?

Checks whether sentence-corpus composition (Switchboard, OpenWebText2,
50-word, frequent-word, random-word, ...) shifts systematically across the
timeline -- a potential confound for the drift analysis, per docs/DATASET.md
section 10.

Produces figures/corpus_composition_over_time.png as a side effect (this
figure IS a committed artifact; this script is its reproducible source --
moved here from scratch/, which is gitignored, after the committed figure
turned out to have no tracked source at all, same problem
tools/measure_padding_waste.py and tools/benchmark_gru.py were moved to
solve).

Run with: uv run tools/investigate_corpus_composition.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("data/external/t15_copyTaskData_description.csv")
MANIFEST_PATH = Path("data/interim/session_manifest.csv")
FIGURES_DIR = Path("figures")

TRAIN_BLOCK_START = "2023-08-11"
TRAIN_BLOCK_END = "2023-10-15"

# Bars narrower than this would touch at the minimum observed inter-session
# gap and become visually indistinguishable from true overlap; this is a
# sanity floor on the derived width, not itself the width used.
MIN_BAR_WIDTH_DAYS = 0.1


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = df.rename(columns={"Number of sentences": "n_trials"})
    return df


def part_a_corpora(df: pd.DataFrame) -> None:
    print("=== Part (a): corpora present and trial counts ===")
    counts = df.groupby("Corpus")["n_trials"].sum().sort_values(ascending=False)
    print(counts.to_string())
    print(f"Total trials across all corpora: {counts.sum()}")
    print(f"Number of distinct corpora: {df['Corpus'].nunique()}")
    known = {"Switchboard", "OpenWebText2", "50-Word", "Freq words", "Random"}
    observed = set(df["Corpus"].unique())
    extra = observed - known
    if extra:
        print(
            f"NOTE: corpora present that docs/DATASET.md section 10 did not "
            f"name: {sorted(extra)} -- DATASET.md should be corrected."
        )


def _bar_width_days(dates: pd.DatetimeIndex) -> float:
    """Derive a stacked-bar width that cannot make adjacent-date bars overlap.

    THE BUG THIS PREVENTS. `part_b_plot_over_time` draws each corpus's
    stacked bars across ALL session dates in one `ax.bar()` call; successive
    calls (one per corpus column) stack via a running `bottom`, but render
    in DRAW order, not x-position order -- a corpus drawn later paints OVER
    anything beneath it wherever its bar's x-extent overlaps. A previous
    version hardcoded `width=3.0` (days). Session 1 (2023-08-11, 0%
    Switchboard) and session 2 (2023-08-13, 84.6% Switchboard) are only 2
    days apart -- less than that width -- so session 2's wide Switchboard
    segment (drawn last: Switchboard sorts alphabetically last among the six
    corpora) overlapped session 1's x-position and painted over it,
    producing a committed figure whose leftmost visible bar read as roughly
    18% 50-Word / 82% Switchboard -- session 2's actual composition, not
    session 1's -- and misled a careful reader into doubting a true claim
    (`docs/FINDINGS.md`: "Per-session Switchboard counts measured directly;
    s20's active-session count corrected from 19 to 17").

    THE FIX. Derive the width from the smallest gap between any two
    consecutive session dates, with a 10% margin, so it is provably smaller
    than every gap -- no two bars can ever touch, regardless of how the
    session calendar is distributed. Verified programmatically, not just by
    construction: every consecutive pair's gap is re-checked against the
    derived width below, and this function raises rather than silently
    returning a width that would reintroduce the bug.

    Args:
        dates: Session dates (one entry per session, need not be sorted or
            unique-by-construction, but duplicate dates are a data error
            this function will reject by producing a zero gap).

    Returns:
        A width in days: 90% of the smallest gap between consecutive
        (sorted) dates.

    Raises:
        AssertionError: If two session dates are identical (zero gap --
            `part_b_plot_over_time`'s groupby should already have merged
            same-date sessions, so this signals a real data problem), or if
            the derived width would not strictly clear every observed gap
            (should be unreachable given how width is computed; checked
            explicitly anyway rather than trusted).
    """
    sorted_dates = sorted(dates)
    gaps = [
        (sorted_dates[i + 1] - sorted_dates[i]).days
        for i in range(len(sorted_dates) - 1)
    ]
    assert min(gaps) > 0, (
        "_bar_width_days: two session dates are identical (zero gap) -- "
        "part_b_plot_over_time's groupby should have merged these; a "
        "duplicate session date is a data problem, not a width problem."
    )
    width = min(gaps) * 0.9
    assert width >= MIN_BAR_WIDTH_DAYS, (
        f"_bar_width_days: derived width {width} days is implausibly "
        f"narrow (min gap {min(gaps)} days) -- check the session date data "
        f"before trusting this figure."
    )
    for i in range(len(sorted_dates) - 1):
        gap = (sorted_dates[i + 1] - sorted_dates[i]).days
        assert gap >= width, (
            f"_bar_width_days: derived width {width} days would overlap "
            f"{sorted_dates[i].date()} and {sorted_dates[i + 1].date()} "
            f"(gap {gap} days) -- this should be unreachable; the width "
            f"formula above is broken if this fires."
        )
    return width


def part_b_plot_over_time(df: pd.DataFrame) -> None:
    print("\n=== Part (b): corpus composition over time ===")
    by_date = (
        df.groupby(["Date", "Corpus"])["n_trials"].sum().unstack(fill_value=0)
    )
    fraction = by_date.div(by_date.sum(axis=1), axis=0)

    # A stacked AREA plot linearly interpolates between session dates,
    # which is misleading here: there are two multi-month recording gaps
    # with no sessions at all, and area interpolation would draw a smooth
    # (fictitious) composition trend across them. Use stacked bars at the
    # true session-date x-position instead, so gaps render as empty space.
    bar_width = _bar_width_days(fraction.index)
    print(f"  Derived bar width: {bar_width:.3f} days (see _bar_width_days)")

    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = pd.Series(0.0, index=fraction.index)
    colors = plt.get_cmap("tab10").colors
    for i, corpus in enumerate(fraction.columns):
        ax.bar(
            fraction.index, fraction[corpus], bottom=bottom, width=bar_width,
            label=corpus, color=colors[i % len(colors)],
        )
        bottom = bottom + fraction[corpus]

    ax.axvspan(pd.Timestamp("2023-08-11"), pd.Timestamp("2023-10-15"),
               color="black", alpha=0.06, zorder=0)
    ax.text(pd.Timestamp("2023-09-13"), 1.02, "planned TRAIN block",
            ha="center", fontsize=8)

    ax.set_ylabel("Fraction of trials that session")
    ax.set_xlabel("Session date")
    ax.set_title("Corpus composition per session over the 20-month recording timeline\n"
                 "(bars at true session dates; empty space = no session that day)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "corpus_composition_over_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


def part_c_train_block_vs_heldout(df: pd.DataFrame) -> None:
    print("\n=== Part (c): planned TRAIN block vs 30 held-out sessions ===")
    train_mask = (df["Date"] >= TRAIN_BLOCK_START) & (df["Date"] <= TRAIN_BLOCK_END)
    train_block = df[train_mask]
    held_out = df[~train_mask]

    print(f"TRAIN block: {train_block['Date'].nunique()} sessions, {train_block['n_trials'].sum()} trials")
    print(f"Held-out:    {held_out['Date'].nunique()} sessions, {held_out['n_trials'].sum()} trials")

    train_frac = train_block.groupby("Corpus")["n_trials"].sum()
    train_frac = (train_frac / train_frac.sum()).sort_values(ascending=False)
    held_frac = held_out.groupby("Corpus")["n_trials"].sum()
    held_frac = (held_frac / held_frac.sum()).sort_values(ascending=False)

    comparison = pd.DataFrame({"train_block_frac": train_frac, "held_out_frac": held_frac}).fillna(0.0)
    comparison["abs_diff"] = (comparison["train_block_frac"] - comparison["held_out_frac"]).abs()
    print(comparison.round(4).to_string())
    print(f"Max absolute fraction difference for any single corpus: {comparison['abs_diff'].max():.4f}")


def part_d_era_concentration(df: pd.DataFrame) -> None:
    print("\n=== Part (d): is any corpus concentrated in one era? ===")
    for corpus, group in df.groupby("Corpus"):
        dates = group["Date"].sort_values()
        span_days = (dates.max() - dates.min()).days
        print(
            f"  {corpus}: {group['n_trials'].sum()} trials, "
            f"active {dates.min().date()} - {dates.max().date()} "
            f"({span_days} day span, {dates.nunique()} distinct sessions)"
        )


def part_e_crosscheck_manifest(df: pd.DataFrame) -> None:
    print("\n=== Part (e): cross-check against data/interim/session_manifest.csv ===")
    manifest = pd.read_csv(MANIFEST_PATH)

    # CSV "Split" is Train vs Val/Test (matches our earlier finding that
    # val and test are trial-level splits of the same held-out blocks).
    # Aggregate the CSV to (session_date, split_group) trial totals.
    df = df.copy()
    df["split_group"] = df["Split"].map({"Train": "train", "Val/Test": "val_test"})
    csv_totals = df.groupby([df["Date"].dt.date.astype(str), "split_group"])["n_trials"].sum()

    manifest["val_test_group"] = manifest["partition"].map(
        {"train": "train", "val": "val_test", "test": "val_test"}
    )
    manifest_totals = manifest.groupby(["session_date", "val_test_group"])["n_trials"].sum()

    mismatches = []
    all_keys = set(csv_totals.index) | set(manifest_totals.index)
    for key in sorted(all_keys):
        csv_n = int(csv_totals.get(key, -1))
        manifest_n = int(manifest_totals.get(key, -1))
        if csv_n != manifest_n:
            mismatches.append((key, csv_n, manifest_n))

    print(f"Compared {len(all_keys)} (session_date, split_group) cells.")
    if mismatches:
        print(f"{len(mismatches)} MISMATCH(ES) found (csv_n vs manifest_n, -1 = missing):")
        for key, csv_n, manifest_n in mismatches:
            print(f"  {key}: csv={csv_n}, manifest={manifest_n}")
    else:
        print("No mismatches: CSV per-session split trial totals agree exactly with the manifest.")

    # Session-date set agreement
    csv_sessions = set(df["Date"].dt.date.astype(str).unique())
    manifest_sessions = set(manifest["session_date"].unique())
    if csv_sessions != manifest_sessions:
        print(f"Session-date sets differ: csv_only={csv_sessions - manifest_sessions}, "
              f"manifest_only={manifest_sessions - csv_sessions}")
    else:
        print(f"Session-date sets agree exactly: {len(csv_sessions)} sessions in both.")


def main() -> None:
    df = load_csv()
    part_a_corpora(df)
    part_b_plot_over_time(df)
    part_c_train_block_vs_heldout(df)
    part_d_era_concentration(df)
    part_e_crosscheck_manifest(df)


if __name__ == "__main__":
    main()
