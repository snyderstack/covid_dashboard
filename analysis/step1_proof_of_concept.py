"""
Step 1 - proof of concept: growth-rate R_eff and full SIR fit for 5 counties.

R_eff is the effective reproduction number at wave onset (see reff_core).

Run from the repo root:
    venv/bin/python analysis/step1_proof_of_concept.py [--gamma-days 7] [--latent-days 3]

Outputs
    analysis/results/step1_reff_poc.csv
    analysis/figures/step1_sir_fits.png / .pdf
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

COUNTIES = [
    # (county name, state, role)
    ("Los Angeles County", "CA", "large urban"),
    ("Maricopa County",    "AZ", "large urban"),
    ("Miami-Dade County",  "FL", "large urban"),
    ("Hidalgo County",     "TX", "large"),
    ("Lancaster County",   "NE", "mid-size"),
]

DISAGREE_FRAC = 0.30   # flag if |Reff_sir - Reff_growth| / Reff_growth exceeds this
POOR_R2 = 0.80         # flag SIR fit with R^2 below this

OBS_COLOR = "#2a78d6"
FIT_COLOR = "#eb6834"
WIN_COLOR = "#cde2fb"
INK = "#0b0b0b"
INK_2 = "#52514e"


def analyze(cases, pop_df, gamma, sigma):
    rows, panels = [], []
    for county, state, role in COUNTIES:
        ts, ma, pop = R.county_case_series(cases, pop_df, county, state)
        wave = R.first_major_wave(ts, ma, pop)
        if wave is None:
            print(f"{county}, {state}: no conservative wave before "
                  f"{R.FIRST_WAVE_CUTOFF.date()} - excluded")
            continue
        g = R.fit_growth_poisson(ts, wave, gamma, sigma)
        s = R.fit_sir(ts, wave, gamma)

        flags = []
        if not g["ok"]:
            flags.append(f"growth fit failed: {g['reason']}")
        rel = abs(s["Reff_sir"] - g["Reff_sir_formula"]) / g["Reff_sir_formula"]
        if rel > DISAGREE_FRAC:
            flags.append(f"R_eff estimates differ by {rel:.0%}")
        if not s["converged"]:
            flags.append("SIR did not converge")
        if s["at_bound"]:
            flags.append(f"SIR parameter at bound: {s['at_bound']}")
        if not (s["r2"] >= POOR_R2):
            flags.append(f"SIR R^2 < {POOR_R2}")

        rows.append({
            "county": county, "state": state, "role": role,
            "population": int(pop), "ma_window_days": ma,
            "smoothing": f"centered {ma}-day MA (lag_analysis.prepare_daily_per_capita)",
            "wave_onset": wave["start_date"].date(),
            "wave_peak": wave["peak_date"].date(),
            "wave_end": wave["end_date"].date(),
            "growth_method": "Poisson GLM, raw daily counts, day-of-week effects",
            "growth_win_start": g["win_start"].date(),
            "growth_win_end": g["win_end"].date(),
            "growth_win_days": g["win_days"],
            "window_cases": g["window_cases"], "dow_effects": g["dow_effects"],
            "r_per_day": g["r"], "r_se": g["r_se"],
            "dispersion_pearson": g["dispersion"], "deviance_df": g["deviance_df"],
            "Reff_growth": g["Reff_sir_formula"], "Reff_growth_se": g["Reff_sir_formula_se"],
            "Reff_seir": g["Reff_seir_formula"], "Reff_seir_se": g["Reff_seir_formula_se"],
            "sir_beta": s["beta"], "sir_beta_se": s["beta_se"],
            "sir_log10_I0": s["log10_I0"], "sir_log10_I0_se": s["log10_I0_se"],
            "sir_reporting_frac": s["rho"], "sir_reporting_frac_se": s["rho_se"],
            "Reff_sir": s["Reff_sir"], "Reff_sir_se": s["Reff_sir_se"],
            "sir_final_attack_rate": s["attack_rate"],
            "sir_r2": s["r2"],
            "sir_converged": s["converged"], "sir_at_bound": s["at_bound"],
            "Reff_rel_diff": rel,
            "flags": "; ".join(flags),
            "gamma_per_day": gamma, "sigma_per_day": sigma,
        })
        panels.append((county, state, ts, wave, g, s))
    return pd.DataFrame(rows), panels


def plot(panels, path_stem):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.35 * nrow), squeeze=False)
    for ax, (county, state, ts, wave, g, s) in zip(axes.flat, panels):
        # Context: 21 days either side of the fitted wave
        lo = max(wave["start_idx"] - 21, 0)
        hi = min(wave["end_idx"] + 21, len(ts) - 1)
        ax.axvspan(g["win_start"], g["win_end"], color=WIN_COLOR, lw=0,
                   label="Growth-fit window")
        ax.plot(ts["Date"].iloc[lo:hi + 1], ts["Per100k MA"].iloc[lo:hi + 1],
                color=OBS_COLOR, lw=1.6, label="Observed (smoothed)")
        ax.plot(s["t_dates"], s["y_fit"], color=FIT_COLOR, lw=1.6, ls="--",
                label="SIR fit")
        ax.set_title(f"{county.replace(' County', '')}, {state}\n"
                     f"$R_{{eff}}^{{SIR}}$ = {s['Reff_sir']:.2f}  "
                     f"($R_{{eff}}^{{growth}}$ = {g['Reff_sir_formula']:.2f})", color=INK)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Daily cases per 100k")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Legend in the spare panel if there is one, else below
    if n < nrow * ncol:
        list(axes.flat)[n].legend(handles, labels, loc="center", frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-days", type=float, default=R.INFECTIOUS_PERIOD_DAYS,
                    help="infectious period 1/gamma in days (default 7)")
    ap.add_argument("--latent-days", type=float, default=R.LATENT_PERIOD_DAYS,
                    help="latent period 1/sigma in days for SEIR R_eff (default 3)")
    args = ap.parse_args()
    gamma, sigma = 1.0 / args.gamma_days, 1.0 / args.latent_days

    os.makedirs(R.FIG_DIR, exist_ok=True)
    os.makedirs(R.RES_DIR, exist_ok=True)
    cases, _deaths, pop_df = R.load_all()
    df, panels = analyze(cases, pop_df, gamma, sigma)

    csv_path = os.path.join(R.RES_DIR, "step1_reff_poc.csv")
    df.to_csv(csv_path, index=False, float_format="%.6g")
    plot(panels, os.path.join(R.FIG_DIR, "step1_sir_fits"))

    show = df[["county", "wave_onset", "wave_peak", "growth_win_start",
               "growth_win_end", "window_cases", "dispersion_pearson",
               "r_per_day", "Reff_growth", "Reff_growth_se",
               "Reff_seir", "Reff_sir", "Reff_sir_se", "sir_r2", "sir_converged",
               "sir_reporting_frac", "sir_reporting_frac_se",
               "sir_final_attack_rate", "flags"]]
    with pd.option_context("display.width", 400, "display.max_columns", 30,
                           "display.max_colwidth", 80):
        print(show.round(4).to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
