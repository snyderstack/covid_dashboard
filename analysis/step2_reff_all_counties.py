"""
Step 2 - growth-rate R_eff for every county with population >= 50,000.

Two datasets:
  MAIN        one row per county (all counties >= 50k, including excluded
              and dropped ones, with a status column). R_eff of the first
              major pre-2021 wave (see reff_core.first_major_wave).
  WAVE-LEVEL  one row per standard-setting wave with onset before 2021-01-01,
              in every county, with an is_main_wave flag.

Growth rate: Poisson GLM on raw daily counts with day-of-week effects over a
fixed 14-day window from the growth start (reff_core.fit_growth_poisson).
Filter (both datasets): dropped only if the window has < 30 cases or the
window (ended at the peak if needed) is < 7 days ("rise too short"). r <= 0 is
kept and flagged (r_nonpositive); R_eff < 1 is allowed. The earlier OLS-on-smoothed
estimate is kept in ols_* columns for comparison; no filter uses it.

Run from the repo root:
    venv/bin/python analysis/step2_reff_all_counties.py [--gamma-days 7] [--latent-days 3]

Outputs
    analysis/results/step2_reff_main.csv
    analysis/results/step2_reff_wave_level.csv
    analysis/results/step2_monthly_median_reff.csv
    analysis/results/step2_metadata.txt
    analysis/figures/step2_reff_histogram.png / .pdf
    analysis/figures/step2_reff_vs_onset.png / .pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

import reff_core as R                      # noqa: E402
from tools import get_population_column    # noqa: E402

MONTHS = pd.period_range("2020-03", "2020-12", freq="M")

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
MAIN_COLOR = "#2a78d6"
OTHER_COLOR = "#a9a8a3"
TREND_COLOR = "#eb6834"


def quality(g):
    """('ok', '') or ('dropped', reason) for a Poisson growth-fit result."""
    return ("ok", "") if g["ok"] else ("dropped", g["reason"])


def growth_cols(g, o):
    """Poisson GLM columns (g) plus the OLS comparison columns (o)."""
    return {
        "growth_win_start": g["win_start"].date(),
        "growth_win_end": g["win_end"].date() if pd.notna(g["win_end"]) else None,
        "growth_win_days": g["win_days"], "window_cases": g["window_cases"],
        "window_truncated_at_peak": g["truncated_at_peak"],
        "dow_effects": g["dow_effects"],
        "r_per_day": g["r"], "r_se": g["r_se"], "r_nonpositive": g["r_nonpositive"],
        "dispersion_pearson": g["dispersion"], "quasi_poisson_se": g["quasi"],
        "deviance_df": g["deviance_df"],
        "Reff_sir_formula": g["Reff_sir_formula"],
        "Reff_sir_formula_se": g["Reff_sir_formula_se"],
        "Reff_seir_formula": g["Reff_seir_formula"],
        "Reff_seir_formula_se": g["Reff_seir_formula_se"],
        "ols_ok": o["ok"], "ols_r_per_day": o["r"], "ols_r_se": o["r_se"],
        "ols_log_r2": o["r2_log"], "ols_Reff_sir_formula": o["Reff_growth"],
        "ols_win_end": o["win_end"].date(), "ols_days_used": o["n_used"],
    }


def analyze(cases, pop_df, gamma, sigma):
    pop_col = get_population_column(pop_df)
    pops = pop_df[pd.to_numeric(pop_df[pop_col], errors="coerce") >= R.MIN_POPULATION]
    main_rows, wave_rows = [], []
    for _, row in pops.iterrows():
        county, state, fips = row["County Name"], row["State"], row["countyFIPS"]
        ts, ma, pop = R.county_case_series(cases, pop_df, county, state, fips=fips)
        ident = {"county": county, "countyFIPS": str(fips).zfill(5), "state": state,
                 "population": int(row[pop_col])}
        waves = R.pre_cutoff_waves(ts, ma, pop)
        k = R.main_wave_index(waves)
        cons = R.has_wave_before(ts, ma, pop, "conservative")

        fits = [(R.fit_growth_poisson(ts, w, gamma, sigma),
                 R.fit_growth_rate(ts, w, pop, gamma, sigma)) for w in waves]
        for i, (w, (g, o)) in enumerate(zip(waves, fits)):
            status, reason = quality(g)
            wave_rows.append({
                **ident, "wave_number": i + 1, "is_main_wave": i == k,
                "wave_onset": w["start_date"].date(), "wave_peak": w["peak_date"].date(),
                "wave_end": w["end_date"].date(),
                "rise_days": w["peak_idx"] - w["start_idx"],
                "peak_height_per100k": w["peak_height"],
                **growth_cols(g, o), "status": status, "drop_reason": reason,
            })

        mrow = {**ident, "rule": f"standard, onset<{R.FIRST_WAVE_CUTOFF.date()}, "
                                 f"peak>={R.MAJOR_PEAK_FRAC:.0%} of max pre-2021 peak",
                "ma_window_days": ma, "conservative_also_found": cons,
                "n_pre2021_waves": len(waves)}
        if k is None:
            # Describe the first standard wave of any date, for the exclusion note
            all_w = R._detect(ts, ma, pop, R.WAVE_SENSITIVITY)
            first = all_w[0] if all_w else None
            mrow.update({
                "status": "excluded",
                "drop_reason": f"no standard wave with onset < {R.FIRST_WAVE_CUTOFF.date()}",
                "first_std_wave_onset": pd.Timestamp(first["start_date"]).date() if first else None,
                "first_std_wave_peak": pd.Timestamp(first["peak_date"]).date() if first else None,
                "max_2020_smoothed_per100k": float(
                    ts.set_index("Date")["Per100k MA"][:"2020-12-31"].max()),
            })
        else:
            w = waves[k]
            g, o = fits[k]
            status, reason = quality(g)
            mrow.update({
                "skipped_small_wave": k > 0, "n_small_waves_skipped": k,
                "wave_onset": w["start_date"].date(), "wave_peak": w["peak_date"].date(),
                "wave_end": w["end_date"].date(),
                "rise_days": w["peak_idx"] - w["start_idx"],
                "peak_height_per100k": w["peak_height"],
                **growth_cols(g, o), "status": status, "drop_reason": reason,
            })
        main_rows.append(mrow)
    return pd.DataFrame(main_rows), pd.DataFrame(wave_rows)


def describe(x):
    x = pd.Series(x).dropna()
    return {"n": len(x), "median": x.median(), "q1": x.quantile(.25),
            "q3": x.quantile(.75), "min": x.min(), "max": x.max()}


def monthly_medians(wl):
    ok = wl[wl.status == "ok"].copy()
    ok["month"] = pd.to_datetime(ok.wave_onset).dt.to_period("M")
    rows = []
    for m in MONTHS:
        sub = ok[ok.month == m]
        rows.append({"onset_month": str(m), "n_waves": len(sub),
                     "median_Reff_sir_formula": sub.Reff_sir_formula.median(),
                     "median_Reff_seir_formula": sub.Reff_seir_formula.median()})
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def plot_histogram(main_ok, gamma_days, stem):
    x = main_ok.Reff_sir_formula.values
    med = float(np.median(x))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bins = np.arange(np.floor(x.min() * 20) / 20, x.max() + 0.05, 0.05)
    ax.hist(x, bins=bins, color=MAIN_COLOR, edgecolor="white", linewidth=0.6)
    ax.axvline(1, color=INK_2, lw=0.8, ls=":")
    ax.axvline(med, color=INK, lw=1.2, ls="--")
    ax.annotate(f"median = {med:.2f}", xy=(med, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top", color=INK)
    ax.set_xlabel(r"$R_{eff}$ at wave onset ($1 + r/\gamma$, Poisson GLM $r$, "
                  rf"$1/\gamma$ = {gamma_days:g} d)")
    ax.set_ylabel("Counties")
    ax.set_title(f"First major wave, counties ≥ 50k population (n = {len(x)})",
                 color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, stem)


def plot_vs_onset(wl_ok, monthly, stem):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    other = wl_ok[~wl_ok.is_main_wave]
    main = wl_ok[wl_ok.is_main_wave]
    ax.scatter(pd.to_datetime(other.wave_onset), other.Reff_sir_formula, s=9,
               color=OTHER_COLOR, alpha=0.6, lw=0, label=f"Other pre-2021 waves (n = {len(other)})")
    ax.scatter(pd.to_datetime(main.wave_onset), main.Reff_sir_formula, s=10,
               color=MAIN_COLOR, alpha=0.75, lw=0, label=f"First major wave (n = {len(main)})")
    mm = monthly.dropna(subset=["median_Reff_sir_formula"])
    mid = pd.to_datetime(mm.onset_month) + pd.Timedelta(days=14)
    ax.plot(mid, mm.median_Reff_sir_formula, color=TREND_COLOR, lw=2,
            marker="o", ms=4, label="Monthly median (all waves)")
    ax.axhline(1, color=INK_2, lw=0.6, ls=":")
    ax.set_ylabel(r"$R_{eff}$ at wave onset ($1 + r/\gamma$, Poisson GLM)")
    ax.set_xlabel("Wave onset date")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp("2020-02-15"), pd.Timestamp("2021-01-01"))
    ax.set_title("Growth-rate $R_{eff}$ by wave onset date, 2020", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, stem)


def write_metadata(path, main, wl, gamma, sigma):
    ex = main[main.status == "excluded"]
    jan2 = ex[pd.to_datetime(ex.first_std_wave_onset).between("2021-01-01", "2021-01-07")]
    jan2_short = jan2[(pd.to_datetime(jan2.first_std_wave_peak)
                       - pd.to_datetime(jan2.first_std_wave_onset)).dt.days <= 2]
    lines = [
        "Step 2 metadata",
        "===============",
        "Quantity: R_eff, effective reproduction number at wave onset, from the",
        "exponential growth rate r.",
        "Growth rate r: Poisson GLM, log link, on RAW daily new case counts",
        "  (prepare_daily_per_capita 'Daily Cases': cumulative differences, negatives",
        "  clipped to 0), log E[cases_t] = a + r*t + day-of-week fixed effects, over a",
        f"  fixed {R.GROWTH_WINDOW_DAYS}-day window from the growth start. Dispersion = Pearson",
        f"  chi2/df; if > {R.OVERDISPERSION_LIMIT:g}, SEs are quasi-Poisson (scale = Pearson chi2/df).",
        "  deviance_df is reported as a goodness-of-fit statistic and not filtered on.",
        f"  SIR formula:  R_eff = 1 + r/gamma,                1/gamma = {1/gamma:g} days",
        f"  SEIR formula: R_eff = (1 + r/sigma)(1 + r/gamma), 1/sigma = {1/sigma:g} days",
        "  SE(R_eff, SIR formula) = SE(r)/gamma; SEIR SE by the delta method.",
        "",
        f"Smoothing: centered {R.MA_WINDOW}-day moving average for all counties "
        "(lag_analysis.prepare_daily_per_capita / wave_analysis.smooth_series).",
        "  The wave detector is run with the same 7-day window. The dashboard itself",
        "  auto-selects 3/5/7/14 days per county; numbers here therefore differ from",
        "  the dashboard's Wave Analysis tab for counties where it picks another window.",
        "",
        "Wave detection: wave_analysis.calculate_wave_metrics, 'standard' preset.",
        f"First major wave (MAIN dataset): earliest standard wave with onset before "
        f"{R.FIRST_WAVE_CUTOFF.date()} whose",
        f"  peak is >= {R.MAJOR_PEAK_FRAC:.0%} of the county's largest pre-2021 standard-wave "
        "peak (peak = smoothed per-100k value on the peak date).",
        "  conservative_also_found: the 'conservative' preset also reports a wave with",
        "  onset before 2021-01-01 (robustness subset for Step 3).",
        "",
        "Growth start: day of the minimum smoothed value within [onset, onset +",
        f"  floor((peak - onset)/2)]. The window ends at the detected peak if the {R.GROWTH_WINDOW_DAYS}-day",
        "  window would run past it (window_truncated_at_peak).",
        f"  Day-of-week effects are used only for windows >= {R.MIN_DOW_DAYS} days (8 parameters);",
        "  shorter windows are fit with the trend only (dow_effects = False).",
        f"Filter: drop only if the window has < {R.MIN_WINDOW_CASES} total cases or < {R.MIN_FIT_DAYS} days",
        "  ('rise too short'). r <= 0 is kept, flagged r_nonpositive; R_eff < 1 allowed.",
        "ols_* columns: earlier estimator (OLS of ln smoothed per-100k series, growth",
        "  start to rise time-midpoint), kept for comparison only; not used to filter.",
        "",
        f"Excluded counties (status = excluded): {len(ex)} with no standard wave starting",
        "  before 2021-01-01; listed in step2_reff_main.csv with the onset and peak of",
        "  their first standard wave of any date and their 2020 maximum smoothed value.",
        f"  {len(jan2_short)} of them have a first standard wave starting 2021-01-01..07 and",
        "  peaking within 2 days of onset; these 1-2 day 'waves' appear to be holiday",
        "  reporting artifacts: "
        + (", ".join(f"{r.county} {r.state}" for r in jan2_short.itertuples()) or "none") + ".",
        "  The rest have a first standard wave starting later in 2021 or after.",
        "",
        "Unit of analysis: county. Associations are ecological.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-days", type=float, default=R.INFECTIOUS_PERIOD_DAYS)
    ap.add_argument("--latent-days", type=float, default=R.LATENT_PERIOD_DAYS)
    args = ap.parse_args()
    gamma, sigma = 1.0 / args.gamma_days, 1.0 / args.latent_days

    os.makedirs(R.FIG_DIR, exist_ok=True)
    os.makedirs(R.RES_DIR, exist_ok=True)
    cases, _deaths, pop_df = R.load_all()
    main_df, wl = analyze(cases, pop_df, gamma, sigma)
    monthly = monthly_medians(wl)

    main_df.to_csv(os.path.join(R.RES_DIR, "step2_reff_main.csv"), index=False,
                   float_format="%.6g")
    wl.to_csv(os.path.join(R.RES_DIR, "step2_reff_wave_level.csv"), index=False,
              float_format="%.6g")
    monthly.to_csv(os.path.join(R.RES_DIR, "step2_monthly_median_reff.csv"),
                   index=False, float_format="%.4g")
    write_metadata(os.path.join(R.RES_DIR, "step2_metadata.txt"), main_df, wl, gamma, sigma)

    style()
    main_ok = main_df[main_df.status == "ok"]
    plot_histogram(main_ok, args.gamma_days, os.path.join(R.FIG_DIR, "step2_reff_histogram"))
    plot_vs_onset(wl[wl.status == "ok"], monthly, os.path.join(R.FIG_DIR, "step2_reff_vs_onset"))

    for name, df in (("MAIN", main_df), ("WAVE-LEVEL", wl)):
        fitted = df[df.status != "excluded"]
        ok = df[df.status == "ok"]
        print(f"\n== {name}: rows {len(df)}, status counts {df.status.value_counts().to_dict()}")
        print("   drop reasons:", df[df.status == "dropped"].drop_reason.value_counts().to_dict())
        for col in ("Reff_sir_formula", "Reff_seir_formula"):
            d = describe(ok[col])
            print(f"   {col}: n={d['n']} median={d['median']:.3f} IQR={d['q1']:.3f}-{d['q3']:.3f} "
                  f"min={d['min']:.3f} max={d['max']:.3f}")
        print(f"   median dispersion (ok): {ok.dispersion_pearson.median():.2f}; "
              f"quasi-Poisson SEs: {int(ok.quasi_poisson_se.sum())}/{len(ok)}; "
              f"median deviance/df: {ok.deviance_df.median():.2f}")
        print(f"   r_nonpositive (ok): {int(ok.r_nonpositive.sum())}; "
              f"window truncated at peak (ok): {int(ok.window_truncated_at_peak.sum())}; "
              f"no day-of-week effects (ok): {int((~ok.dow_effects.astype(bool)).sum())}")
        terc = pd.qcut(fitted.population, 3)
        print("   pass rate by population tercile:",
              {str(k): round(v, 3) for k, v in
               fitted.groupby(terc, observed=True).status.apply(lambda x: (x == "ok").mean()).items()})
        if "conservative_also_found" in fitted:
            print("   pass rate by conservative_also_found:",
                  fitted.groupby("conservative_also_found").status
                        .apply(lambda x: round((x == "ok").mean(), 3)).to_dict())
        both = fitted[fitted.ols_ok & fitted.r_per_day.notna()]
        print(f"   Poisson r vs OLS r (n={len(both)} with both): "
              f"Pearson {both.r_per_day.corr(both.ols_r_per_day):.3f}, "
              f"Spearman {both.r_per_day.corr(both.ols_r_per_day, method='spearman'):.3f}, "
              f"median difference (Poisson - OLS) {(both.r_per_day - both.ols_r_per_day).median():.4f}")
    print("\nmonthly medians (wave-level, by onset month):")
    print(monthly.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
