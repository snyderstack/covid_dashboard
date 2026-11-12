"""
Step 5 - stress tests of the Step 3 (R_eff) and Step 4 (deaths per case)
trait associations.

Every model keeps state-clustered SEs and traits in SD units (z-scored within
each model's sample). p_normal is the statsmodels normal-based p-value; p_t
uses a t distribution with (clusters - 1) df; sig_bonf_t = p_t < 0.05/6.

R_eff (WLS, weights 1/SE^2 unless noted; onset-month FE unless noted)
  main                       single trait (= Step 3 main table)
  all_six                    all six traits in one model
  state_fe                   single trait + state FE (margin dropped: constant within state)
  all_six_state_fe           five traits + state FE
  half_month_fe              onset half-month FE (days 1-15 / 16-end) instead of month FE
  iso_week_fe                onset ISO-week FE instead of month FE
  re_weights                 weights 1/(SE^2 + tau^2), tau^2 by method of moments
  re_weights_all_six           (DerSimonian-Laird for meta-regression, from the
                               fixed-weight fit of the same design)
  unweighted                 OLS
  no_dow                     growth GLM refit without day-of-week terms
  window_10/21/28            growth GLM refit on a 10/21/28-day window (10-day windows
                             have no day-of-week terms: they need >= 12 days)
  ols_smoothed               earlier estimator: OLS of ln smoothed per-100k series
                             (Step 2 ols_* columns), weights 1/SE(OLS)^2
  first_wave                 very first 2020 wave (peak_frac 0)
  largest_wave               largest 2020 wave (peak_frac 1)
  winsorized_se_le1          counties with SE(R_eff) <= 1, R_eff winsorized at
                             their 1st/99th percentiles
  onset_from_may             wave onset >= 2020-05-01
Deaths per case (quasi-Poisson, offset log cases, deaths lagged 14 days)
  main, all_six, state_fe, all_six_state_fe, half_month_fe,
  cases_per_person_control, cases_per_person_control_all_six
                             + log(cases_wave / population) as a control
  onset_from_may, onset_from_may_all_six
  first_wave, largest_wave   cases and deaths recomputed for that wave

Samples: every variant starts from the Step 3 / Step 4 county sample (Step 2
status ok; Step 4 also cases_wave >= 50). Refit variants keep the counties
whose refit is ok (Step 2 rules) and, for deaths per case, cases_wave >= 50.

Influence (step5_influence.csv): for every R_eff version above plus the Step 3
robustness versions (conservative subset, smallest tercile removed,
r_nonpositive removed, weights capped at the 95th percentile): the largest
single-county share of the total weight, and the estimate and p after
dropping each of the five highest-weight counties one at a time. sig_flip
marks a change in sig_bonf_t.

Estimates: R_eff rows are the change in R_eff per 1 SD; deaths-per-case rows
are the log rate-ratio per 1 SD (100*(exp(estimate)-1) = % change), with se
on the same scale.

Run from the repo root (after steps 2-4):
    venv/bin/python analysis/step5_stress_tests.py [--compare path/to/stress_test_results.csv]

Outputs
    analysis/results_v2/step5_stress_tests.csv
    analysis/results_v2/step5_refit_table.csv      per-county refits used above
    analysis/results_v2/step5_influence.csv        top-5 weight drop-one check
    analysis/results_v2/step5_compare.csv          only with --compare
    analysis/figures_v2/step5_stress_tests.png / .pdf
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import statsmodels.api as sm               # noqa: E402

import reff_core as R                      # noqa: E402
import step3_reff_vs_factors as S3         # noqa: E402
import step4_deaths_per_case as S4         # noqa: E402

FACTORS = S3.FACTORS
TRAITS = [x for x, *_ in FACTORS]
LABELS = {x: label for x, label, _ in FACTORS}
STATE_LEVEL = "pres2020_margin_d"
BONF = 0.05 / len(TRAITS)
LAG = S4.LAG
MAY = pd.Timestamp("2020-05-01")
WINDOWS = (10, 21, 28)
NO_DOW = 10 ** 9                           # min_dow_days that no window reaches

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DOT = "#2a78d6"


# ---- per-county refits -------------------------------------------------------

def refit_table(d):
    """
    For every county in the Step 3 sample: growth refits of its first major
    wave (no day-of-week terms; 10/21/28-day windows), and growth fit plus
    cases / lagged deaths for its very first and its largest pre-2021 wave.
    """
    cases_df, deaths_df, pop_df = R.load_all()
    rows = []
    for rec in d.itertuples():
        ts, ma, pop = R.county_case_series(cases_df, pop_df, rec.county, rec.state,
                                           fips=rec.countyFIPS)
        waves = R.pre_cutoff_waves(ts, ma, pop)
        k = R.main_wave_index(waves)
        out = {"countyFIPS": rec.countyFIPS}
        variants = {"no_dow": R.fit_growth_poisson(ts, waves[k], min_dow_days=NO_DOW)}
        for W in WINDOWS:
            variants[f"window_{W}"] = R.fit_growth_poisson(ts, waves[k], window_days=W)
        for name, g in variants.items():
            out[f"{name}_r"] = g["r"] if g["ok"] else np.nan
            out[f"{name}_r_se"] = g["r_se"] if g["ok"] else np.nan
        c = S4._daily(cases_df, pop_df, rec.countyFIPS, rec.state, "Cases")
        dd = S4._daily(deaths_df, pop_df, rec.countyFIPS, rec.state, "Deaths")
        for name, j in (("first_wave", 0), ("largest_wave", R.main_wave_index(waves, 1.0))):
            w = waves[j]
            g = R.fit_growth_poisson(ts, w)
            onset, end = w["start_date"], w["end_date"]
            lo, hi = onset + pd.Timedelta(days=LAG), end + pd.Timedelta(days=LAG)
            out.update({
                f"{name}_number": j + 1, f"{name}_onset": onset.date(),
                f"{name}_r": g["r"] if g["ok"] else np.nan,
                f"{name}_r_se": g["r_se"] if g["ok"] else np.nan,
                f"{name}_cases": float(c[onset:end].sum()),
                f"{name}_deaths_lag{LAG}": float(dd[lo:hi].sum()) if dd is not None else np.nan,
            })
        rows.append(out)
    return pd.DataFrame(rows)


# ---- model helpers -----------------------------------------------------------

def add_period_keys(d, onset_col="wave_onset"):
    on = pd.to_datetime(d[onset_col])
    d = d.copy()
    d["onset_month"] = on.dt.to_period("M").astype(str)
    d["onset_half_month"] = d.onset_month + np.where(on.dt.day <= 15, "a", "b")
    iso = on.dt.isocalendar()
    d["onset_iso_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return d


def design(sub, traits, fe, controls=()):
    z = sub[traits]
    parts = [(z - z.mean()) / z.std(ddof=1)]
    for col in fe:
        parts.append(pd.get_dummies(sub[col], prefix=col, drop_first=True, dtype=float))
    if controls:
        parts.append(sub[list(controls)])
    return sm.add_constant(pd.concat(parts, axis=1), has_constant="add")


def rows_from(res, sub, traits, outcome, model):
    G = sub.state.nunique()
    out = []
    for x in traits:
        p_t, *_ = R.t_inference(res.params[x], res.bse[x], G)
        out.append({"outcome": outcome, "model": model, "trait": x,
                    "estimate": res.params[x], "se": res.bse[x],
                    "p_normal": res.pvalues[x], "p_t": p_t, "n": len(sub),
                    "sig_bonf_t": p_t < BONF})
    return out


def fit_reff(d, traits, y="Reff", se="Reff_se", fe=("onset_month",),
             weights="inverse_variance"):
    """WLS of R_eff on z-scored traits + FE, state-clustered SEs.
    Returns (results, sample, weights)."""
    sub = d.dropna(subset=[y, se] + list(traits))
    sub = sub[sub[se] > 0]
    X = design(sub, list(traits), fe)
    v = sub[se] ** 2
    if weights == "unweighted":
        w = pd.Series(1.0, index=sub.index)
    elif weights == "random_effects":
        w = 1 / (v + R.tau2_mm(sub[y], X, v))
    elif weights == "capped95":
        w = (1 / v).clip(upper=(1 / v).quantile(0.95))
    else:
        w = 1 / v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.WLS(sub[y], X, weights=w).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.state)[0]})
    return res, sub, w


def reff_model(d, traits, model, **kw):
    res, sub, _ = fit_reff(d, traits, **kw)
    return rows_from(res, sub, traits, "Reff", model)


def severity_model(d, traits, model, y=f"deaths_lag{LAG}", cases="cases_wave",
                   fe=("onset_month",), controls=()):
    sub = d.dropna(subset=[y, cases] + list(traits) + list(controls))
    sub = sub[sub[cases] >= S4.MIN_CASES]
    X = design(sub, list(traits), fe, controls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(sub[y], X, family=sm.families.Poisson(),
                     offset=np.log(sub[cases])).fit(
            scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.state)[0]})
    return rows_from(res, sub, traits, "deaths_per_case", model)


def each(fn, d, model, **kw):
    """Single-trait models, one per trait."""
    return [r for x in TRAITS for r in fn(d, [x], model, **kw)]


def each_within_state(fn, d, model, **kw):
    return [r for x in TRAITS if x != STATE_LEVEL
            for r in fn(d, [x], model, fe=("onset_month", "state"), **kw)]


# ---- variants ----------------------------------------------------------------

def reff_specs(d, rf):
    """
    Every R_eff model version as (model, data, traits, fit_reff kwargs);
    single-trait versions contribute one spec per trait.
    """
    g = R.INFECTIOUS_PERIOD_DAYS
    d = add_period_keys(d)
    within = [x for x in TRAITS if x != STATE_LEVEL]
    specs = []

    def each_trait(model, data, traits=TRAITS, **kw):
        specs.extend((model, data, [x], kw) for x in traits)

    each_trait("main", d)
    specs.append(("all_six", d, TRAITS, {}))
    each_trait("state_fe", d, within, fe=("onset_month", "state"))
    specs.append(("all_six_state_fe", d, within, {"fe": ("onset_month", "state")}))
    each_trait("half_month_fe", d, fe=("onset_half_month",))
    each_trait("iso_week_fe", d, fe=("onset_iso_week",))
    each_trait("re_weights", d, weights="random_effects")
    specs.append(("re_weights_all_six", d, TRAITS, {"weights": "random_effects"}))
    each_trait("unweighted", d, weights="unweighted")

    m = d.merge(rf, on="countyFIPS", how="left")
    for name in ("no_dow", *[f"window_{W}" for W in WINDOWS]):
        each_trait(name, m.assign(Reff=1 + m[f"{name}_r"] * g, Reff_se=m[f"{name}_r_se"] * g))
    each_trait("ols_smoothed", d[d.ols_ok.astype(bool)].assign(
        Reff=lambda t: 1 + t.ols_r_per_day * g, Reff_se=lambda t: t.ols_r_se * g))
    for name in ("first_wave", "largest_wave"):
        v = m.assign(Reff=1 + m[f"{name}_r"] * g, Reff_se=m[f"{name}_r_se"] * g,
                     wave_onset=m[f"{name}_onset"])
        each_trait(name, add_period_keys(v))

    w = d[d.Reff_se <= 1].copy()
    lo, hi = w.Reff.quantile([0.01, 0.99])
    w["Reff"] = w.Reff.clip(lo, hi)
    each_trait("winsorized_se_le1", w)
    each_trait("onset_from_may", d[pd.to_datetime(d.wave_onset) >= MAY])
    return specs


def step3_robustness_specs(d):
    """Step 3 robustness / weight-robustness versions not in the stress table
    (influence check only)."""
    d = add_period_keys(d)
    specs = []
    for model, data, kw in [
            ("s3_conservative_also_found", d[d.conservative_also_found.astype(bool)], {}),
            ("s3_excl_smallest_pop_tercile", d[d.pop_tercile > 1], {}),
            ("s3_excl_r_nonpositive", d[~d.r_nonpositive.astype(bool)], {}),
            ("s3_weights_capped95", d, {"weights": "capped95"})]:
        specs.extend((model, data, [x], kw) for x in TRAITS)
    return specs


def reff_variants(d, rf):
    return [r for model, data, traits, kw in reff_specs(d, rf)
            for r in reff_model(data, traits, model, **kw)]


def influence(specs, n_drop=5):
    """
    For each R_eff model: largest single-county share of the total weight, and
    the estimate and p after dropping each of the n_drop highest-weight
    counties one at a time (weights, incl. tau^2, recomputed after the drop).
    One row per (model, trait, dropped county).
    """
    rows = []
    for model, data, traits, kw in specs:
        res, sub, w = fit_reff(data, traits, **kw)
        G = sub.state.nunique()
        share = (w / w.sum()).sort_values(ascending=False)
        full = {x: (res.params[x], res.pvalues[x],
                    R.t_inference(res.params[x], res.bse[x], G)[0]) for x in traits}
        for rank, (ix, sh) in enumerate(share.head(n_drop).items(), 1):
            fips = sub.at[ix, "countyFIPS"]
            r2, s2, _ = fit_reff(data[data.countyFIPS != fips], traits, **kw)
            G2 = s2.state.nunique()
            for x in traits:
                est, p, p_t = full[x]
                p_t2 = R.t_inference(r2.params[x], r2.bse[x], G2)[0]
                rows.append({
                    "model": model, "trait": x, "n": len(sub),
                    "max_weight_share": share.iloc[0],
                    "max_weight_county": f"{sub.at[share.index[0], 'county']}, "
                                         f"{sub.at[share.index[0], 'state']}",
                    "estimate": est, "p_normal": p, "p_t": p_t, "sig_bonf_t": p_t < BONF,
                    "drop_rank": rank, "dropped_fips": fips,
                    "dropped_county": f"{sub.at[ix, 'county']}, {sub.at[ix, 'state']}",
                    "dropped_weight_share": sh,
                    "estimate_drop": r2.params[x], "p_normal_drop": r2.pvalues[x],
                    "p_t_drop": p_t2, "sig_bonf_t_drop": p_t2 < BONF,
                    "sig_flip": (p_t < BONF) != (p_t2 < BONF),
                })
    return pd.DataFrame(rows)


def severity_variants(d, rf):
    y = f"deaths_lag{LAG}"
    d = add_period_keys(d)
    d["log_cases_per_person"] = np.log(d.cases_wave / d.population)
    within = [x for x in TRAITS if x != STATE_LEVEL]
    may = d[pd.to_datetime(d.wave_onset) >= MAY]
    out = []
    out += each(severity_model, d, "main")
    out += severity_model(d, TRAITS, "all_six")
    out += each_within_state(severity_model, d, "state_fe")
    out += severity_model(d, within, "all_six_state_fe", fe=("onset_month", "state"))
    out += each(severity_model, d, "half_month_fe", fe=("onset_half_month",))
    out += each(severity_model, d, "cases_per_person_control",
                controls=("log_cases_per_person",))
    out += severity_model(d, TRAITS, "cases_per_person_control_all_six",
                          controls=("log_cases_per_person",))
    out += each(severity_model, may, "onset_from_may")
    out += severity_model(may, TRAITS, "onset_from_may_all_six")
    m = d.merge(rf, on="countyFIPS", how="left")
    for name in ("first_wave", "largest_wave"):
        v = m.assign(**{"cases_wave": m[f"{name}_cases"], y: m[f"{name}_deaths_lag{LAG}"],
                        "wave_onset": m[f"{name}_onset"]})
        out += each(severity_model, add_period_keys(v), name)
    return out


# ---- comparison and figure -------------------------------------------------------

def compare(new, path, threshold=0.10):
    """Rows whose estimate differs from the reference file by > threshold
    (relative to the reference). Matched on outcome, model and trait."""
    old = pd.read_csv(path)
    m = new.merge(old, on=["outcome", "model", "trait"], how="outer",
                  suffixes=("", "_ref"), indicator=True)
    m["rel_diff"] = (m.estimate - m.estimate_ref).abs() / m.estimate_ref.abs()
    m["differs_gt_10pct"] = m.rel_diff > threshold
    return m


def figure(t, stem):
    order = list(dict.fromkeys(t.model))
    panels = [("Reff", "Change in $R_{eff}$ per 1 SD", lambda v: v),
              ("deaths_per_case", "% change in deaths per case per 1 SD",
               lambda v: 100 * np.expm1(v))]
    fig, axes = plt.subplots(len(TRAITS), 2, figsize=(7.4, 8.2), sharex="col")
    for j, (outcome, xlabel, tf) in enumerate(panels):
        models = [mo for mo in order if mo in set(t[t.outcome == outcome].model)]
        offs = dict(zip(models, np.linspace(-0.32, 0.32, len(models))))
        for i, x in enumerate(TRAITS):
            ax = axes[i, j]
            sub = t[(t.outcome == outcome) & (t.trait == x)]
            ax.axvline(0, color=INK_2, lw=0.8, ls=":")
            for r in sub.itertuples():
                main = r.model == "main"
                ax.scatter(tf(r.estimate), offs[r.model], s=34 if main else 16,
                           marker="D" if main else "o",
                           facecolor=(INK if main else DOT) if r.sig_bonf_t else "white",
                           edgecolor=INK if main else DOT, lw=1.0, zorder=3 if main else 2)
            if sub.empty:
                ax.text(0.5, 0, "not estimable (state-level)", transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=7, color=INK_2)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.grid(axis="x", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("left",):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(LABELS[x].replace(" (", "\n("), rotation=0, ha="right",
                              va="center", fontsize=7.5)
        axes[0, j].set_title("$R_{eff}$ (WLS)" if outcome == "Reff"
                             else "Deaths per case (quasi-Poisson)", color=INK)
        axes[-1, j].set_xlabel(xlabel)
    handles = [
        Line2D([], [], ls="", marker="D", ms=6, mfc=INK, mec=INK, label="main model"),
        Line2D([], [], ls="", marker="o", ms=5, mfc=DOT, mec=DOT,
               label=f"p_t < {BONF:.4f} (Bonferroni, t with clusters - 1 df)"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=DOT, label="not significant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    n_r = t[t.outcome == "Reff"].model.nunique()
    n_s = t[t.outcome == "deaths_per_case"].model.nunique()
    R.write_caption(stem, (
        "Stress tests of the trait associations. One row per trait; each dot is one model "
        f"version ({n_r} for R_eff, left; {n_s} for deaths per case, right), placed at its "
        "estimate per 1 SD of the trait (dots are spread vertically within a row only so "
        "they do not overlap). Filled: p < 0.05/6 with a t distribution with (clusters - 1) "
        "df and state-clustered SEs; hollow: not significant. Diamond: main model. The "
        "state-level presidential margin cannot be estimated with state fixed effects. "
        "Versions are listed in step5_stress_tests.py and step5_stress_tests.csv."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", help="earlier stress_test_results.csv to compare against")
    args = ap.parse_args()
    os.makedirs(R.FIG_DIR, exist_ok=True)

    d3 = pd.read_csv(os.path.join(R.RES_DIR, "step3_analysis_table.csv"),
                     dtype={"countyFIPS": str})
    d4 = pd.read_csv(os.path.join(R.RES_DIR, "step4_analysis_table.csv"),
                     dtype={"countyFIPS": str})
    rf_path = os.path.join(R.RES_DIR, "step5_refit_table.csv")
    rf = refit_table(d3)
    rf.to_csv(rf_path, index=False, float_format="%.6g")

    t = pd.DataFrame(reff_variants(d3, rf) + severity_variants(d4, rf))
    infl = influence(reff_specs(d3, rf) + step3_robustness_specs(d3))
    infl.to_csv(os.path.join(R.RES_DIR, "step5_influence.csv"), index=False,
                float_format="%.6g")
    t.to_csv(os.path.join(R.RES_DIR, "step5_stress_tests.csv"), index=False,
             float_format="%.6g")
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    figure(t, os.path.join(R.FIG_DIR, "step5_stress_tests"))

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    flips = infl[infl.sig_flip]
    print(f"\ninfluence: {infl[['model', 'trait']].drop_duplicates().shape[0]} model x trait "
          f"estimates; largest single-county weight share "
          f"{infl.max_weight_share.max():.1%}; Bonferroni flips after one drop: {len(flips)}")
    if len(flips):
        print(flips[["model", "trait", "dropped_county", "dropped_weight_share", "p_t",
                     "p_t_drop", "estimate", "estimate_drop"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.compare:
        c = compare(t, args.compare)
        c.to_csv(os.path.join(R.RES_DIR, "step5_compare.csv"), index=False,
                 float_format="%.6g")
        print("\nonly in one file:")
        print(c[c._merge != "both"][["outcome", "model", "trait", "_merge"]].to_string(index=False))
        print("\nestimates differing by > 10%:")
        print(c[c.differs_gt_10pct][["outcome", "model", "trait", "estimate", "estimate_ref",
                                     "rel_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
