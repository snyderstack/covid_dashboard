"""
Step 3 - R_eff (growth rate, SIR formula) vs six pre-chosen county factors.

Uses the MAIN dataset from Step 2 (analysis/results/step2_reff_main.csv,
status == "ok") joined to the dashboard's master county table
(county_features.create_master_county_table with ahrf_loader's AHRF table)
and state political data (tools.load_state_political).

Two factors are not in the master table and are read directly from
data/ahrf2023.csv by FIPS:
  * % uninsured: pers_noins_lt65_pct_20 (% of persons under 65 without
    health insurance, 2020)
  * poverty rate: pers_povty_pct_20 (% of persons in poverty, 2020), used
    because median HOUSEHOLD income is not in the AHRF file (only median
    family income is)

Per factor:
  a) Spearman rho, p, n (R_eff vs raw factor, unadjusted)
  b) WLS: R_eff ~ z(factor) + onset-month fixed effects,
     weights 1/SE(R_eff)^2, SEs clustered by state. z is standardized
     within each model's sample. Coefficient = change in R_eff per 1 SD.
Bonferroni threshold 0.05/6.

Robustness (WLS only): conservative_also_found subset; smallest population
tercile removed; r_nonpositive waves removed. Weight robustness: unweighted,
and weights capped at their 95th percentile. (gamma is not varied: R_eff - 1
= r/gamma scales by the same factor for every county, so WLS p-values are
invariant to gamma.)

Variance decomposition (weighted R^2, same weights): month FE only; month +
all six factors (with a state-clustered joint Wald F test that the six factor
coefficients are zero); month + state FE (presidential margin omitted: it is
constant within state).

Run from the repo root (after step2):
    venv/bin/python analysis/step3_reff_vs_factors.py

Outputs
    analysis/results/step3_analysis_table.csv
    analysis/results/step3_main_results.csv
    analysis/results/step3_robustness.csv
    analysis/results/step3_weight_robustness.csv
    analysis/results/step3_variance_decomposition.csv
    analysis/results/step3_notes.txt
    analysis/figures/step3_uninsured_onset_confounding.png / .pdf
    analysis/figures/step3_factor_scatter.png / .pdf
    analysis/figures/step3_factor_scatter_month_adjusted.png / .pdf
"""

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
from ahrf_loader import build_ahrf_feature_table      # noqa: E402
from county_features import create_master_county_table  # noqa: E402
from tools import DATA_DIR, load_state_political      # noqa: E402

N_TESTS = 6
ALPHA = 0.05
BONF = ALPHA / N_TESTS

FACTORS = [
    # (column, label for tables, axis label)
    ("log10_pop_density", "log10 population density (per sq mi)", "log10 population density (per sq mi)"),
    ("pcp_per_100k", "Primary care physicians per 100k", "Primary care physicians per 100k"),
    ("pct_pop_65plus", "% aged 65+", "% aged 65+"),
    ("poverty_pct", "Poverty rate (%)", "Poverty rate (%)"),
    ("uninsured_lt65_pct", "% uninsured (under 65)", "% uninsured (under 65)"),
    ("pres2020_margin_d", "2020 presidential margin, D-R (state)", "2020 pres. margin D-R (pp, state)"),
]

Y = "Reff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
DOT = "#2a78d6"
LINE = "#eb6834"


# ---- data ------------------------------------------------------------------

def load_factors(cases, deaths, pop_df):
    ahrf, _ = build_ahrf_feature_table(covid_fips=set(cases["countyFIPS"]), verbose=False)
    master, _ = create_master_county_table(cases, deaths, pop_df, ahrf_df=ahrf, verbose=False)
    master["countyFIPS"] = master["countyFIPS"].astype(str).str.zfill(5)

    extra = pd.read_csv(DATA_DIR / "ahrf2023.csv",
                        usecols=["fips_st_cnty", "pers_noins_lt65_pct_20", "pers_povty_pct_20"],
                        dtype={"fips_st_cnty": str}, encoding="latin-1")
    extra["countyFIPS"] = extra["fips_st_cnty"].str.zfill(5)
    extra = extra.rename(columns={"pers_noins_lt65_pct_20": "uninsured_lt65_pct",
                                  "pers_povty_pct_20": "poverty_pct"})
    for c in ("uninsured_lt65_pct", "poverty_pct"):
        extra[c] = pd.to_numeric(extra[c], errors="coerce")

    pol = load_state_political()[["state_abbr", "pres_2020_margin_d"]].rename(
        columns={"state_abbr": "state", "pres_2020_margin_d": "pres2020_margin_d"})

    f = master[["countyFIPS", "pop_density_per_sqmi", "pcp_per_100k", "pct_pop_65plus",
                "icu_beds_per_100k", "snf_beds_per_100k"]].merge(
        extra[["countyFIPS", "uninsured_lt65_pct", "poverty_pct"]], on="countyFIPS", how="left")
    dens = pd.to_numeric(f["pop_density_per_sqmi"], errors="coerce")
    f["log10_pop_density"] = np.where(dens > 0, np.log10(dens.where(dens > 0)), np.nan)
    return f, pol


def build_table(gamma_days=R.INFECTIOUS_PERIOD_DAYS):
    cases, deaths, pop_df = R.load_all()
    main = pd.read_csv(os.path.join(R.RES_DIR, "step2_reff_main.csv"),
                       dtype={"countyFIPS": str})
    fitted = main[main.status != "excluded"].copy()
    # population tercile among all fitted counties (as reported in Step 2)
    fitted["pop_tercile"] = pd.qcut(fitted.population, 3, labels=[1, 2, 3]).astype(int)
    d = fitted[fitted.status == "ok"].copy()
    d["onset_month"] = pd.to_datetime(d.wave_onset).dt.to_period("M").astype(str)
    d["r_nonpositive"] = d["r_nonpositive"].astype(bool)
    d["conservative_also_found"] = d["conservative_also_found"].astype(bool)
    f, pol = load_factors(cases, deaths, pop_df)
    d = d.merge(f, on="countyFIPS", how="left").merge(pol, on="state", how="left")
    return d


def with_gamma(d, gamma_days):
    """R_eff = 1 + r/gamma and SE = SE(r)/gamma for a given infectious period."""
    out = d.copy()
    out[Y] = 1 + out.r_per_day * gamma_days
    out["Reff_se"] = out.r_se * gamma_days
    return out


# ---- models ----------------------------------------------------------------

def spearman(d, x):
    sub = d[[Y, x]].dropna()
    rho, p = stats.spearmanr(sub[Y], sub[x])
    return rho, p, len(sub)


def _design(sub, x, month_fe=True, standardize=True):
    cols = []
    if x is not None:
        v = sub[x]
        cols.append(((v - v.mean()) / v.std(ddof=1)).rename("z") if standardize else v.rename("z"))
    if month_fe:
        cols.append(pd.get_dummies(sub.onset_month, prefix="m", drop_first=True, dtype=float))
    X = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=sub.index)
    return sm.add_constant(X, has_constant="add")


def weights_for(sub, scheme="inverse_variance"):
    """1/SE^2 ("inverse_variance"), the same capped at its 95th percentile
    ("capped95"), or all ones ("unweighted")."""
    w = 1 / sub.Reff_se ** 2
    if scheme == "capped95":
        return w.clip(upper=w.quantile(0.95))
    if scheme == "unweighted":
        return pd.Series(1.0, index=sub.index)
    return w


def wls(d, x, month_fe=True, standardize=True, weights="inverse_variance"):
    """WLS of R_eff on (z-scored) x [+ month FE], state-clustered SEs."""
    sub = d.dropna(subset=[Y, "Reff_se", x]).copy()
    sub = sub[sub.Reff_se > 0]
    X = _design(sub, x, month_fe, standardize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.WLS(sub[Y], X, weights=weights_for(sub, weights)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.state)[0]})
    lo, hi = res.conf_int().loc["z"]
    return {"coef": res.params["z"], "se": res.bse["z"], "p": res.pvalues["z"],
            "ci_lo": lo, "ci_hi": hi,
            "n": len(sub), "n_states": sub.state.nunique(), "res": res, "sub": sub}


def month_adjusted(d):
    """Residuals of R_eff from a WLS month-only model."""
    sub = d.dropna(subset=[Y, "Reff_se"]).copy()
    sub = sub[sub.Reff_se > 0]
    X = _design(sub, None, month_fe=True)
    res = sm.WLS(sub[Y], X, weights=1 / sub.Reff_se ** 2).fit()
    out = d.copy()
    out["Reff_month_adj"] = np.nan
    out.loc[sub.index, "Reff_month_adj"] = sub[Y] - res.fittedvalues
    return out


# ---- outputs ---------------------------------------------------------------

def main_results(d):
    rows = []
    for x, label, _ in FACTORS:
        rho, p_s, n_s = spearman(d, x)
        w = wls(d, x)
        rows.append({"factor": label, "column": x,
                     "spearman_rho": rho, "spearman_p": p_s, "spearman_n": n_s,
                     "spearman_sig_bonf": p_s < BONF,
                     "wls_coef_per_sd": w["coef"], "wls_se": w["se"],
                     "wls_ci95_lo": w["ci_lo"], "wls_ci95_hi": w["ci_hi"], "wls_p": w["p"],
                     "wls_n": w["n"], "wls_n_states": w["n_states"],
                     "wls_sig_bonf": w["p"] < BONF})
    return pd.DataFrame(rows)


def robustness(d):
    variants = [
        ("main", d, "inverse_variance"),
        ("(i) conservative_also_found", d[d.conservative_also_found], "inverse_variance"),
        ("(ii) excl. smallest pop tercile", d[d.pop_tercile > 1], "inverse_variance"),
        ("(iii) excl. r_nonpositive", d[~d.r_nonpositive], "inverse_variance"),
    ]
    return _variant_table(variants)


def weight_robustness(d):
    return _variant_table([
        ("main (weights 1/SE^2)", d, "inverse_variance"),
        ("(a) unweighted", d, "unweighted"),
        ("(b) weights capped at 95th pct", d, "capped95"),
    ])


def _variant_table(variants):
    rows = []
    for name, dd, scheme in variants:
        for x, label, _ in FACTORS:
            w = wls(dd, x, weights=scheme)
            rows.append({"variant": name, "factor": label, "coef_per_sd": w["coef"],
                         "se": w["se"], "ci95_lo": w["ci_lo"], "ci95_hi": w["ci_hi"],
                         "p": w["p"], "n": w["n"],
                         "sig_bonf": w["p"] < BONF})
    return pd.DataFrame(rows)


def consistency_notes(rob):
    notes = {}
    for label in rob.factor.unique():
        r = rob[rob.factor == label]
        signs = set(np.sign(r.coef_per_sd))
        sig = set(r.sig_bonf)
        base = r.iloc[0]
        sign_txt = ("same sign in all variants" if len(signs) == 1
                    else "sign differs across variants: " + ", ".join(
                        f"{v} {'+' if c > 0 else '-'}" for v, c in zip(r.variant, r.coef_per_sd)))
        if len(sig) == 1:
            sig_txt = ("significant after Bonferroni in all variants" if base.sig_bonf
                       else "not significant after Bonferroni in any variant")
        else:
            sig_txt = ("Bonferroni-significant in: "
                       + ", ".join(r.variant[r.sig_bonf]) + "; not in: "
                       + ", ".join(r.variant[~r.sig_bonf]))
        notes[label] = f"{sign_txt}; {sig_txt}."
    return notes


def weighted_r2(res, y, w):
    """Weighted R^2 = 1 - sum w e^2 / sum w (y - ybar_w)^2."""
    ybar = np.sum(w * y) / np.sum(w)
    return 1 - np.sum(w * res.resid ** 2) / np.sum(w * (y - ybar) ** 2)


def variance_decomposition(d):
    cols = [x for x, _, _ in FACTORS]
    sub = d.dropna(subset=[Y, "Reff_se"] + cols).copy()
    sub = sub[sub.Reff_se > 0]
    w = weights_for(sub)
    groups = pd.factorize(sub.state)[0]
    months = pd.get_dummies(sub.onset_month, prefix="m", drop_first=True, dtype=float)
    z = (sub[cols] - sub[cols].mean()) / sub[cols].std(ddof=1)
    states = pd.get_dummies(sub.state, prefix="st", drop_first=True, dtype=float)
    designs = {
        "(a) onset-month FE only": months,
        "(b) month FE + six factors": pd.concat([months, z], axis=1),
        "(c) month FE + state FE": pd.concat([months, states], axis=1),
    }
    rows, joint = [], None
    for name, X in designs.items():
        X = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.WLS(sub[Y], X, weights=w).fit(
                cov_type="cluster", cov_kwds={"groups": groups})
        rows.append({"model": name, "weighted_r2": weighted_r2(res, sub[Y], w),
                     "n_params": X.shape[1], "n": len(sub)})
        if name.startswith("(b)"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ft = res.f_test(", ".join(f"{c} = 0" for c in cols))
            joint = {"F": float(np.squeeze(ft.fvalue)), "df_num": int(ft.df_num),
                     "df_denom": int(ft.df_denom), "p": float(ft.pvalue)}
    return pd.DataFrame(rows), joint


def uninsured_figure(d, stem):
    x, xlabel = "uninsured_lt65_pct", "% uninsured (under 65, 2020)"
    sub = d.dropna(subset=[Y, "Reff_se", x, "Reff_month_adj"])
    sub = sub[sub.Reff_se > 0]
    w = 1 / sub.Reff_se ** 2
    size = 2 + 120 * w / w.max()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    panels = [(Y, r"$R_{eff}$ (growth rate, $1/\gamma$ = 7 d)", "Unadjusted"),
              ("Reff_month_adj", r"Month-adjusted $R_{eff}$ (residual)",
               "Adjusted for onset month")]
    for ax, (ycol, ylabel, title) in zip(axes, panels):
        ax.scatter(sub[x], sub[ycol], s=size, color=DOT, alpha=0.35, lw=0)
        fit = sm.WLS(sub[ycol], sm.add_constant(sub[x]), weights=w).fit()
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, color=LINE, lw=1.8)
        ax.set_title(f"{title}: slope {fit.params.iloc[1]:+.4f} per pp", color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        lo, hi = np.nanpercentile(sub[ycol], [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "Point area proportional to WLS weight 1/SE²; lines: WLS fits "
             "(no clustering). y-axis: central 99% of values.",
             ha="center", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def write_notes(path, joint):
    lines = [
        "Step 3 notes",
        "============",
        "- R_eff = 1 + r/gamma, so R_eff - 1 scales by 1/gamma for every county; WLS",
        "  coefficients scale with 1/gamma and p-values are invariant to gamma.",
        "  gamma is therefore not varied in the robustness tables.",
        "- Poverty rate (AHRF pers_povty_pct_20) is used because median household income",
        "  is not in the AHRF file (only median family income).",
        "- % uninsured is AHRF pers_noins_lt65_pct_20: % of persons under 65 without",
        "  health insurance, 2020 (SAHIE).",
        "- Population density is AHRF/master-table pop_density_per_sqmi (2020 Census",
        "  population / 2020 land area), log10-transformed.",
        "- Presidential margin is state-level (tools.load_state_political), shared by all",
        "  counties in a state; SEs are clustered by state (51 clusters incl. DC).",
        "- Weighted R^2 = 1 - sum w e^2 / sum w (y - weighted mean)^2.",
        f"- Joint test (model b, state-clustered): F({joint['df_num']}, {joint['df_denom']}) = "
        f"{joint['F']:.3f}, p = {joint['p']:.4g}.",
        "- Associations are county-level (ecological).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def scatter_figure(d, ycol, ylabel, stem, results, line_note, titles="spearman"):
    """titles: "spearman" (unadjusted Spearman for raw R_eff) or "wls" (the
    month-adjusted WLS result from the main table)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey=True)
    for ax, (x, _label, xlabel) in zip(axes.flat, FACTORS):
        sub = d.dropna(subset=[ycol, "Reff_se", x])
        sub = sub[sub.Reff_se > 0]
        w = 1 / sub.Reff_se ** 2
        size = 2 + 120 * w / w.max()          # marker area proportional to weight
        ax.scatter(sub[x], sub[ycol], s=size, color=DOT, alpha=0.35, lw=0)
        # WLS fit line, raw factor scale, no month effects
        X = sm.add_constant(sub[x])
        fit = sm.WLS(sub[ycol], X, weights=w).fit()
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, fit.params.iloc[0] + fit.params.iloc[1] * xs, color=LINE, lw=1.8)
        row = results.set_index("column").loc[x]
        if titles == "wls":
            ax.set_title(f"coef per SD = {row.wls_coef_per_sd:+.3f}, p = {row.wls_p:.2g}",
                         color=INK, fontsize=8)
        else:
            ax.set_title(f"ρ = {row.spearman_rho:.2f}, p = {row.spearman_p:.2g} "
                         f"(n = {int(row.spearman_n)})", color=INK)
        ax.set_xlabel(xlabel)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    lo, hi = np.nanpercentile(d[ycol], [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    axes.flat[0].set_ylim(lo - pad, hi + pad)
    title_note = ("Titles: month-adjusted WLS coefficient per SD and p (state-clustered SEs), "
                  "as in the Step 3 main table." if titles == "wls"
                  else "Titles: unadjusted Spearman for raw R_eff.")
    fig.text(0.5, 0.005, "Point area proportional to WLS weight 1/SE²; " + line_note
             + "\n" + title_note + " y-axis: central 99% of values.",
             ha="center", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    os.makedirs(R.FIG_DIR, exist_ok=True)
    d = build_table()
    d = with_gamma(d, R.INFECTIOUS_PERIOD_DAYS)
    d = month_adjusted(d)
    d.to_csv(os.path.join(R.RES_DIR, "step3_analysis_table.csv"), index=False,
             float_format="%.6g")

    res = main_results(d)
    res.to_csv(os.path.join(R.RES_DIR, "step3_main_results.csv"), index=False,
               float_format="%.6g")
    rob = robustness(d)
    rob.to_csv(os.path.join(R.RES_DIR, "step3_robustness.csv"), index=False,
               float_format="%.6g")
    wrob = weight_robustness(d)
    wrob.to_csv(os.path.join(R.RES_DIR, "step3_weight_robustness.csv"), index=False,
                float_format="%.6g")
    vd, joint = variance_decomposition(d)
    vd.to_csv(os.path.join(R.RES_DIR, "step3_variance_decomposition.csv"), index=False,
              float_format="%.6g")
    write_notes(os.path.join(R.RES_DIR, "step3_notes.txt"), joint)

    style()
    scatter_figure(d, Y, r"$R_{eff}$ (growth rate, $1/\gamma$ = 7 d)",
                   os.path.join(R.FIG_DIR, "step3_factor_scatter"), res,
                   "line: WLS fit without month effects.")
    scatter_figure(d, "Reff_month_adj", r"Month-adjusted $R_{eff}$ (residual)",
                   os.path.join(R.FIG_DIR, "step3_factor_scatter_month_adjusted"), res,
                   "line: WLS fit of month-adjusted R_eff on the factor.", titles="wls")
    R.write_caption(os.path.join(R.FIG_DIR, "step3_factor_scatter_month_adjusted"), (
        "Month-adjusted R_eff (residual from a WLS model of R_eff on onset-month fixed "
        f"effects, weights 1/SE²) against each of the six county factors, {len(d)} counties. "
        "Point area is proportional to the weight; line: WLS fit of the residual on the raw "
        "factor. Panel titles give the month-adjusted WLS result for that factor from the "
        "Step 3 main table: change in R_eff per 1 SD of the factor (R_eff ~ z(factor) + "
        "onset-month fixed effects, weights 1/SE²) and its p-value with SEs clustered by "
        f"state. Bonferroni threshold for six tests: p < {BONF:.4f}. y-axis: central 99% "
        "of values."))
    uninsured_figure(d, os.path.join(R.FIG_DIR, "step3_uninsured_onset_confounding"))

    w = 1 / d.Reff_se ** 2
    top = w.sort_values(ascending=False)
    print(f"analysis sample: {len(d)} counties, {d.state.nunique()} states; "
          f"Bonferroni threshold {BONF:.4f}")
    print(f"weight share of top 10 / top 50 counties: {top.head(10).sum() / w.sum():.1%} / "
          f"{top.head(50).sum() / w.sum():.1%}")
    print("missing factor values:", {x: int(d[x].isna().sum()) for x, _, _ in FACTORS})
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(res.drop(columns="column").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
        print(rob.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
        print(wrob.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
        print(vd.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("joint test (model b):", joint)
    print()
    print("robustness consistency:")
    for k, v in consistency_notes(rob).items():
        print(f"- {k}: {v}")
    print("weight-robustness consistency:")
    for k, v in consistency_notes(wrob).items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
