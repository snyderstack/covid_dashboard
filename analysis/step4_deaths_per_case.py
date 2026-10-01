"""
Step 4 - deaths per case during each county's first major pre-2021 wave vs
county factors.

Sample: the Step 3 MAIN dataset (step2_reff_main.csv, status == "ok"), same
wave per county. Factors come from step3_reff_vs_factors.load_factors (the
dashboard master table + AHRF columns read by FIPS + state political data).

Outcome
  cases_wave  = raw daily new cases summed from wave onset to wave end
  deaths_wave = raw daily new deaths summed over [onset + LAG, end + LAG]
  Daily series: lag_analysis.prepare_daily_per_capita(ma_window=1), i.e.
  cumulative differences with negatives clipped to 0 (the dashboard pipeline).
  Counties with zero deaths are kept; counties with cases_wave < MIN_CASES
  are dropped.
  Secondary outcome: deaths_wave per 100k (population offset).
Overlap flag
  death_window_overlap: the shifted death window [.., end + LAG] reaches the
  next detected standard wave's onset + LAG (equivalently end >= next onset).
  death_window_overlap_strict (reported only): end + LAG >= next onset.

Primary model
  Poisson GLM, log link, offset log(cases_wave), onset-month fixed effects +
  one standardized factor; quasi-Poisson scale (Pearson chi2/df); SEs
  clustered by state. Effect = 100 * (exp(b) - 1) % change in deaths per
  case per 1 SD.

Run from the repo root (after step2 and step3):
    venv/bin/python analysis/step4_deaths_per_case.py [--lag 14]
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import statsmodels.api as sm               # noqa: E402
from scipy import stats                    # noqa: E402

import reff_core as R                      # noqa: E402
import step3_reff_vs_factors as S3         # noqa: E402
from lag_analysis import prepare_daily_per_capita  # noqa: E402

LAG = 14
ROBUST_LAGS = (7, 21)
MIN_CASES = 50
SCARCE_TESTING_END = pd.Timestamp("2020-05-01")

FACTORS = S3.FACTORS
BONF = 0.05 / len(FACTORS)
SECONDARY = [
    ("icu_beds_per_100k", "ICU beds per 100k",
     "AHRF stgh_med_surg_icu_beds_21 (med-surg ICU beds, short-term general "
     "hospitals, 2021) per 100k 2020 Census population (master table)"),
    ("snf_beds_per_100k", "SNF beds per 100k",
     "AHRF snf_beds_21 (skilled nursing facility beds, 2021) per 100k 2020 "
     "Census population (master table)"),
]
BONF_SECONDARY = 0.05 / len(SECONDARY)
MONTHS = pd.period_range("2020-03", "2020-12", freq="M")

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DOT, LINE = "#2a78d6", "#eb6834"


# ---- data ------------------------------------------------------------------

def _daily(df_wide, pop_df, fips, state, metric):
    """Raw daily counts (negatives clipped), indexed by date; name found by FIPS."""
    row = df_wide[(df_wide["countyFIPS"] == fips) & (df_wide["State"] == state)]
    if row.empty:
        return None
    ts = prepare_daily_per_capita(df_wide, pop_df, row.iloc[0]["County Name"], state,
                                  metric, ma_window=1, fips=fips)
    return ts.set_index("Date")[f"Daily {metric}"].astype(float)


def build_data(lags):
    cases_df, deaths_df, pop_df = R.load_all()
    d = S3.with_gamma(S3.build_table(), R.INFECTIOUS_PERIOD_DAYS)
    rows = []
    for rec in d.itertuples():
        fips, state = rec.countyFIPS, rec.state
        c = _daily(cases_df, pop_df, fips, state, "Cases")
        dd = _daily(deaths_df, pop_df, fips, state, "Deaths")
        onset, end = pd.Timestamp(rec.wave_onset), pd.Timestamp(rec.wave_end)
        out = {"countyFIPS": fips, "cases_wave": float(c[onset:end].sum())}
        for L in lags:
            lo, hi = onset + pd.Timedelta(days=L), end + pd.Timedelta(days=L)
            out[f"deaths_lag{L}"] = float(dd[lo:hi].sum()) if dd is not None else np.nan
        # next detected standard wave (any date) after this one
        ts, ma, pop = R.county_case_series(cases_df, pop_df, rec.county, state, fips=fips)
        later = [pd.Timestamp(w["start_date"]) for w in R._detect(ts, ma, pop, R.WAVE_SENSITIVITY)
                 if pd.Timestamp(w["start_date"]) > onset]
        nxt = min(later) if later else pd.NaT
        out["next_wave_onset"] = nxt.date() if pd.notna(nxt) else None
        out["death_window_overlap"] = bool(pd.notna(nxt) and end >= nxt)
        for L in lags:
            out[f"death_window_overlap_strict_lag{L}"] = bool(
                pd.notna(nxt) and end + pd.Timedelta(days=L) >= nxt)
        rows.append(out)
    d = d.merge(pd.DataFrame(rows), on="countyFIPS", how="left")
    for L in lags:
        d[f"dpc_lag{L}"] = d[f"deaths_lag{L}"] / d.cases_wave
        d[f"deaths_per100k_lag{L}"] = d[f"deaths_lag{L}"] / d.population * 1e5
    return d


# ---- models ----------------------------------------------------------------

def _X(sub, xcols, month_fe=True, state_fe=False):
    parts = []
    if xcols:
        z = sub[xcols]
        parts.append((z - z.mean()) / z.std(ddof=1))
    if month_fe:
        parts.append(pd.get_dummies(sub.onset_month, prefix="m", drop_first=True, dtype=float))
    if state_fe:
        parts.append(pd.get_dummies(sub.state, prefix="st", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sub.index)
    return sm.add_constant(X, has_constant="add")


def fit(d, xcols, y, offset_col, family="qp", month_fe=True, state_fe=False):
    sub = d.dropna(subset=[y, offset_col] + list(xcols)).copy()
    sub = sub[sub[offset_col] > 0]
    X = _X(sub, list(xcols), month_fe, state_fe)
    groups = pd.factorize(sub.state)[0]
    off = np.log(sub[offset_col])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "nb":
            res = sm.NegativeBinomial(sub[y], X, offset=off).fit(
                cov_type="cluster", cov_kwds={"groups": groups}, disp=0, maxiter=500)
            disp = np.nan
        else:
            pois = sm.GLM(sub[y], X, family=sm.families.Poisson(), offset=off)
            res0 = pois.fit()
            disp = float(res0.pearson_chi2 / res0.df_resid)
            res = pois.fit(scale="X2", cov_type="cluster", cov_kwds={"groups": groups})
    return res, sub, disp


def effect(res, col, n, disp=np.nan):
    b = res.params[col]
    lo, hi = res.conf_int().loc[col]
    return {"pct_change_per_sd": 100 * np.expm1(b), "ci95_lo": 100 * np.expm1(lo),
            "ci95_hi": 100 * np.expm1(hi), "p": res.pvalues[col], "log_coef": b,
            "log_se": res.bse[col], "n": n, "dispersion": disp}


def factor_table(d, factors, y, offset_col, bonf, family="qp", label=""):
    rows = []
    for x, name, *_ in factors:
        res, sub, disp = fit(d, [x], y, offset_col, family)
        e = effect(res, x, len(sub), disp)
        rows.append({"variant": label, "factor": name, "column": x, **e,
                     "sig_bonf": e["p"] < bonf})
    return pd.DataFrame(rows)


def decomposition(d, y, offset_col):
    cols = [x for x, *_ in FACTORS]
    sub = d.dropna(subset=[y, offset_col] + cols).copy()
    off = np.log(sub[offset_col])
    groups = pd.factorize(sub.state)[0]
    null = sm.GLM(sub[y], np.ones((len(sub), 1)), family=sm.families.Poisson(),
                  offset=off).fit()
    specs = {
        "(a) onset-month FE only": _X(sub, [], True),
        "(b) month FE + six factors": _X(sub, cols, True),
        "(c) month FE + state FE": _X(sub, [], True, state_fe=True),
    }
    rows, joint = [], None
    for name, X in specs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(sub[y], X, family=sm.families.Poisson(), offset=off)
            res = model.fit()
            rows.append({"model": name, "deviance_explained": 1 - res.deviance / null.deviance,
                         "n_params": X.shape[1], "n": len(sub)})
            if name.startswith("(b)"):
                resc = model.fit(scale="X2", cov_type="cluster", cov_kwds={"groups": groups})
                ft = resc.f_test(", ".join(f"{c} = 0" for c in cols))
                joint = {"F": float(np.squeeze(ft.fvalue)), "df_num": int(ft.df_num),
                         "df_denom": int(ft.df_denom), "p": float(ft.pvalue)}
    return pd.DataFrame(rows), joint


def robustness(d, lag):
    y, off = f"deaths_lag{lag}", "cases_wave"
    base = [
        ("main (LAG 14, quasi-Poisson)", d, y, "qp"),
        ("(i) LAG = 7", d, f"deaths_lag{ROBUST_LAGS[0]}", "qp"),
        ("(ii) LAG = 21", d, f"deaths_lag{ROBUST_LAGS[1]}", "qp"),
        ("(iii) conservative_also_found", d[d.conservative_also_found], y, "qp"),
        ("(iv) excl. smallest pop tercile", d[d.pop_tercile > 1], y, "qp"),
        ("(v) excl. onset < 2020-05-01",
         d[pd.to_datetime(d.wave_onset) >= SCARCE_TESTING_END], y, "qp"),
        ("(vi) negative binomial", d, y, "nb"),
        ("(vii) excl. death-window overlap", d[~d.death_window_overlap], y, "qp"),
        ("(vii-b) excl. strict overlap (end+LAG >= next onset)",
         d[~d[f"death_window_overlap_strict_lag{lag}"]], y, "qp"),
    ]
    return pd.concat([factor_table(dd, FACTORS, yy, off, BONF, fam, name)
                      for name, dd, yy, fam in base], ignore_index=True)


def consistency(rob):
    notes = {}
    for f in rob.factor.unique():
        r = rob[rob.factor == f]
        signs = set(np.sign(r.pct_change_per_sd))
        sig = r.sig_bonf
        s = ("same sign in all variants" if len(signs) == 1 else
             "sign differs: " + ", ".join(f"{v} {'+' if c > 0 else '-'}"
                                         for v, c in zip(r.variant, r.pct_change_per_sd)))
        if sig.all():
            t = "Bonferroni-significant in all variants"
        elif not sig.any():
            t = "not Bonferroni-significant in any variant"
        else:
            t = ("Bonferroni-significant in " + ", ".join(r.variant[sig])
                 + "; not in " + ", ".join(r.variant[~sig]))
        notes[f] = f"{s}; {t}."
    return notes


def pooled_by_month(d, y):
    m = pd.to_datetime(d.wave_onset).dt.to_period("M")
    rows = [{"onset_month": "all", "n_counties": len(d), "deaths": d[y].sum(),
             "cases": d.cases_wave.sum(), "deaths_per_case": d[y].sum() / d.cases_wave.sum()}]
    for mo in MONTHS:
        sub = d[m == mo]
        rows.append({"onset_month": str(mo), "n_counties": len(sub), "deaths": sub[y].sum(),
                     "cases": sub.cases_wave.sum(),
                     "deaths_per_case": sub[y].sum() / sub.cases_wave.sum()
                     if len(sub) else np.nan})
    return pd.DataFrame(rows)


# ---- figures ---------------------------------------------------------------

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def forest(s3, s4, stem):
    labels = [name for _, name, _ in FACTORS]
    s3 = s3.set_index("factor").loc[labels]
    s4 = s4.set_index("factor").loc[labels]
    ypos = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    panels = [
        (axes[0], s3.wls_coef_per_sd, s3.wls_ci95_lo, s3.wls_ci95_hi,
         "Step 3: $R_{eff}$\n(change per 1 SD)"),
        (axes[1], s4.pct_change_per_sd, s4.ci95_lo, s4.ci95_hi,
         "Step 4: deaths per case\n(% change per 1 SD)"),
    ]
    for ax, est, lo, hi, title in panels:
        ax.axvline(0, color=INK_2, lw=0.8, ls=":")
        ax.errorbar(est, ypos, xerr=[est - lo, hi - est], fmt="o", color=DOT,
                    ecolor=DOT, elinewidth=1.6, capsize=0, ms=5)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels)
    fig.text(0.5, 0.005, "Points: estimates; bars: 95% CI (unadjusted for multiple testing). "
             "Both models include onset-month fixed effects; SEs clustered by state.",
             ha="center", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, stem)


def pooled_figure(pooled, stem):
    p = pooled[pooled.onset_month != "all"].dropna(subset=["deaths_per_case"])
    overall = pooled.loc[pooled.onset_month == "all", "deaths_per_case"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = pd.to_datetime(p.onset_month)
    ax.plot(x, 100 * p.deaths_per_case, color=DOT, lw=2, marker="o", ms=5)
    for xi, yi, n in zip(x, 100 * p.deaths_per_case, p.n_counties):
        ax.annotate(f"n={n}", (xi, yi), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=INK_2)
    ax.axhline(100 * overall, color=INK_2, lw=0.8, ls="--")
    ax.annotate(f"all waves: {100 * overall:.2f}%", (x.iloc[1], 100 * overall),
                xytext=(0, -11), textcoords="offset points", ha="left", fontsize=7.5,
                color=INK_2)
    ax.set_ylabel(f"Pooled deaths per case (%), lag {LAG} d")
    ax.set_xlabel("Wave onset month (2020)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Sum of deaths / sum of cases, first major wave", color=INK)
    fig.tight_layout()
    save(fig, stem)


def age_scatter(d, y, stem):
    sub = d.dropna(subset=[y, "pct_pop_65plus"]).copy()
    X = _X(sub, [], True)
    off = np.log(sub.cases_wave)
    month = sm.GLM(sub[y], X, family=sm.families.Poisson(), offset=off).fit()
    sub["expected"] = month.fittedvalues
    sub["oe"] = sub[y] / sub.expected
    curve = sm.GLM(sub[y], sm.add_constant(sub.pct_pop_65plus), family=sm.families.Poisson(),
                   offset=np.log(sub.expected)).fit()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    size = 2 + 150 * sub.cases_wave / sub.cases_wave.max()
    ax.scatter(sub.pct_pop_65plus, sub.oe, s=size, color=DOT, alpha=0.35, lw=0)
    xs = np.linspace(sub.pct_pop_65plus.min(), sub.pct_pop_65plus.max(), 60)
    ax.plot(xs, np.exp(curve.params.iloc[0] + curve.params.iloc[1] * xs), color=LINE, lw=1.8)
    ax.axhline(1, color=INK_2, lw=0.8, ls=":")
    ax.set_ylim(0, np.nanpercentile(sub.oe, 99) * 1.05)
    ax.set_xlabel("% aged 65+")
    ax.set_ylabel("Deaths per case, month-adjusted\n(observed / expected)")
    ax.set_title("Deaths per case vs % aged 65+", color=INK)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "Expected = Poisson model with onset-month effects and log(cases) offset. "
             "Point area proportional to cases in the wave; line: Poisson fit. "
             "y-axis: up to the 99th percentile.",
             ha="center", va="bottom", fontsize=6.3, color=INK_2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, stem)


# ---- main ------------------------------------------------------------------

def main():
    global LAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=LAG)
    LAG = ap.parse_args().lag
    lags = sorted({LAG, *ROBUST_LAGS})
    y = f"deaths_lag{LAG}"

    d_all = build_data(lags)
    n_small = int((d_all.cases_wave < MIN_CASES).sum())
    d = d_all[d_all.cases_wave >= MIN_CASES].copy()
    d.to_csv(os.path.join(R.RES_DIR, "step4_analysis_table.csv"), index=False,
             float_format="%.6g")

    main_t = factor_table(d, FACTORS, y, "cases_wave", BONF, label="main")
    sec_t = factor_table(d, SECONDARY, y, "cases_wave", BONF_SECONDARY, label="secondary")
    # deaths per 100k: death count with log(population) offset
    p100k = factor_table(d, FACTORS, y, "population", BONF, label="deaths per 100k")
    decomp, joint = decomposition(d, y, "cases_wave")
    rob = robustness(d, LAG)
    pooled = pooled_by_month(d, y)
    both = d.dropna(subset=["Reff", f"dpc_lag{LAG}"])
    rho, rho_p = stats.spearmanr(both.Reff, both[f"dpc_lag{LAG}"])

    for name, t in [("step4_main_results", main_t), ("step4_secondary_factors", sec_t),
                    ("step4_deaths_per100k", p100k), ("step4_decomposition", decomp),
                    ("step4_robustness", rob), ("step4_pooled_by_month", pooled)]:
        t.to_csv(os.path.join(R.RES_DIR, f"{name}.csv"), index=False, float_format="%.6g")

    with open(os.path.join(R.RES_DIR, "step4_notes.txt"), "w") as f:
        f.write("\n".join([
            "Step 4 notes", "============",
            f"Outcome: deaths summed over [onset + {LAG}, end + {LAG}] days / cases summed over "
            "[onset, end]; raw daily counts from lag_analysis.prepare_daily_per_capita",
            "  (cumulative differences, negatives clipped to 0). Zero-death counties kept.",
            f"Dropped: cases_wave < {MIN_CASES}: {n_small} counties.",
            "Model: Poisson GLM, log link, offset log(cases_wave), onset-month FE + one "
            "standardized factor; quasi-Poisson scale (Pearson chi2/df); state-clustered SEs.",
            "  Effect: 100*(exp(b)-1) % change in deaths per case per 1 SD.",
            "Secondary factors (Bonferroni 0.05/2):",
            *[f"  {name}: {src}" for _, name, src in SECONDARY],
            "Deaths per 100k: same model with deaths as outcome and log(population) offset "
            "(USAFacts population).",
            "death_window_overlap: end + LAG >= next standard-wave onset + LAG (i.e. end >= "
            "next onset).",
            "death_window_overlap_strict_lagL: end + L >= next standard-wave onset.",
            f"Joint test, model (b): F({joint['df_num']}, {joint['df_denom']}) = {joint['F']:.3f}, "
            f"p = {joint['p']:.4g}.",
            "Associations are county-level (ecological).",
        ]) + "\n")

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    s3 = pd.read_csv(os.path.join(R.RES_DIR, "step3_main_results.csv"))
    forest(s3, main_t, os.path.join(R.FIG_DIR, "step4_forest_reff_vs_dpc"))
    pooled_figure(pooled, os.path.join(R.FIG_DIR, "step4_pooled_dpc_by_month"))
    age_scatter(d, y, os.path.join(R.FIG_DIR, "step4_dpc_vs_age65_month_adjusted"))

    fmt = lambda v: f"{v:.4g}"                     # noqa: E731
    print(f"counties in Step 3 MAIN: {len(d_all)}; dropped cases_wave < {MIN_CASES}: {n_small}; "
          f"analysed: {len(d)}; zero-death counties: {int((d[y] == 0).sum())}")
    print(f"overlap flag (end >= next onset): {int(d.death_window_overlap.sum())}; "
          f"strict (end+{LAG} >= next onset): {int(d[f'death_window_overlap_strict_lag{LAG}'].sum())}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        for t in (main_t, sec_t, p100k, decomp, rob, pooled):
            print(t.drop(columns=[c for c in ("column",) if c in t]).to_string(index=False, float_format=fmt))
            print()
    print("joint test:", joint)
    print(f"Spearman R_eff vs deaths per case: rho={rho:.4f} p={rho_p:.4g} n={len(both)}")
    for k, v in consistency(rob).items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
