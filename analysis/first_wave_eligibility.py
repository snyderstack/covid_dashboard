"""
Apply the first-major-wave rule (see reff_core: earliest standard-setting wave
with onset before 2021-01-01 whose peak is >= 50% of the largest pre-2021
standard-setting peak) to every county with population >= 50,000.

Run from the repo root:
    venv/bin/python analysis/first_wave_eligibility.py [--peak-frac 0.5]

Outputs
    analysis/results/first_wave_eligibility.csv          (one row per county)
    analysis/results/first_wave_exclusions_by_state.csv

Exit status 2 if any state has more than STATE_STOP_PCT % of its counties
excluded (the stop condition agreed for the analysis).
"""

import argparse
import os
import sys

import pandas as pd

import reff_core as R
from tools import get_population_column

STATE_STOP_PCT = 30.0


def build_table(cases, pop_df, peak_frac=R.MAJOR_PEAK_FRAC, ma_window=R.MA_WINDOW):
    """One row per county >= MIN_POPULATION with the selected first wave."""
    pop_col = get_population_column(pop_df)
    pops = pop_df[pd.to_numeric(pop_df[pop_col], errors="coerce") >= R.MIN_POPULATION]
    rows = []
    for _, row in pops.iterrows():
        county, state, fips = row["County Name"], row["State"], row["countyFIPS"]
        ts, ma, pop = R.county_case_series(cases, pop_df, county, state, fips=fips,
                                           ma_window=ma_window)
        wave = None if ts is None else R.first_major_wave(ts, ma, pop, peak_frac=peak_frac)
        cons = False if ts is None else R.has_wave_before(ts, ma, pop, "conservative")
        rows.append({
            "countyFIPS": str(fips).zfill(5), "county": county, "state": state,
            "population": int(row[pop_col]),
            "ma_window_days": ma,
            "has_case_series": ts is not None,
            "eligible": wave is not None,
            "wave_onset": wave["start_date"].date() if wave else None,
            "wave_peak": wave["peak_date"].date() if wave else None,
            "wave_end": wave["end_date"].date() if wave else None,
            "peak_height_per100k": wave["peak_height"] if wave else None,
            "max_pre2021_peak_per100k": wave["max_pre_cutoff_peak"] if wave else None,
            "n_pre2021_waves": wave["n_pre_cutoff_waves"] if wave else 0,
            "n_small_waves_skipped": wave["n_skipped_small"] if wave else 0,
            "skipped_small_wave": bool(wave and wave["n_skipped_small"] > 0),
            "conservative_also_found": cons,
        })
    return pd.DataFrame(rows)


def by_state_table(df):
    return (df.groupby("state")
              .agg(counties=("eligible", "size"),
                   excluded=("eligible", lambda s: int((~s).sum())))
              .assign(excluded_pct=lambda d: 100 * d.excluded / d.counties)
              .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-frac", type=float, default=R.MAJOR_PEAK_FRAC)
    args = ap.parse_args()

    os.makedirs(R.RES_DIR, exist_ok=True)
    cases, _deaths, pop_df = R.load_all()
    df = build_table(cases, pop_df, peak_frac=args.peak_frac)
    df.to_csv(os.path.join(R.RES_DIR, "first_wave_eligibility.csv"),
              index=False, float_format="%.6g")
    bs = by_state_table(df)
    bs.to_csv(os.path.join(R.RES_DIR, "first_wave_exclusions_by_state.csv"),
              index=False, float_format="%.1f")

    n, n_ex = len(df), int((~df.eligible).sum())
    el = df[df.eligible]
    print(f"counties with population >= {R.MIN_POPULATION:,}: {n}")
    print(f"no case series: {int((~df.has_case_series).sum())}")
    print(f"excluded (no standard wave with onset < {R.FIRST_WAVE_CUTOFF.date()}): "
          f"{n_ex} ({100 * n_ex / n:.1f}%)")
    print(f"eligible counties where an earlier small wave was skipped "
          f"(peak < {args.peak_frac:.0%} of largest pre-2021 peak): "
          f"{int(el.skipped_small_wave.sum())} of {len(el)}")
    print(f"eligible counties where conservative also found a pre-2021 wave: "
          f"{int(el.conservative_also_found.sum())} of {len(el)}")
    print("MA window days (eligible):", el.ma_window_days.value_counts().sort_index().to_dict())
    print(bs[bs.excluded > 0].sort_values(["excluded_pct", "excluded"], ascending=False)
            .to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    over = bs[bs.excluded_pct > STATE_STOP_PCT]
    if not over.empty:
        print(f"\nSTOP: states over {STATE_STOP_PCT:.0f}% excluded: "
              + ", ".join(f"{r.state} {r.excluded}/{r.counties}" for r in over.itertuples()))
        sys.exit(2)


if __name__ == "__main__":
    main()
