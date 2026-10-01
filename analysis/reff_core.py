"""
Shared R_eff estimation code for the thesis analysis scripts.

The quantity estimated is R_eff, the effective reproduction number at wave
onset (not the basic reproduction number R0: by a county's first major wave
some immunity and behavioral change may already exist).

Every series and wave boundary comes from the dashboard's own modules, so
numbers match what the dashboard shows:

  * daily cases per 100k and its smoothed version ("Per100k MA")
        lag_analysis.prepare_daily_per_capita
    Smoothing is wave_analysis.smooth_series: a CENTERED moving average
    (7 days here, see MA_WINDOW)
    (np.convolve mode="same"), with the incomplete-window days at each end set
    to NaN. Both the growth fit and the SIR fit use this "Per100k MA" column.
  * smoothing window (auto, as on the Wave Analysis tab)
        wave_analysis.estimate_optimal_smoothing
  * wave onset / peak / end
        wave_analysis.calculate_wave_metrics (region-based detector)

This module never modifies dashboard code; it only imports it.

Definitions
-----------
First major wave
    Among the waves the detector reports at WAVE_SENSITIVITY ("standard")
    with onset before FIRST_WAVE_CUTOFF (2021-01-01), the earliest one whose
    peak is >= MAJOR_PEAK_FRAC (0.5) x the largest of their peaks. Peak
    height is the smoothed per-100k value on the peak date. Counties with no
    standard wave starting before the cutoff are excluded.
Smoothing window
    Centered 7-day moving average for every county (MA_WINDOW = 7), applied
    with the dashboard's own prepare_daily_per_capita / smooth_series. The
    dashboard itself auto-selects 3, 5, 7 or 14 days per county
    (estimate_optimal_smoothing); set MA_WINDOW = None to reproduce that.
    The wave detector is run with the same window.
Growth start
    The day of minimum smoothed value within [onset, onset + floor((peak -
    onset) / 2)] (the first half of the rise), so a flat stretch after the
    detected onset is skipped.
Growth rate r - primary estimator (fit_growth_poisson)
    Poisson GLM, log link, on RAW daily new case counts (the "Daily Cases"
    column of prepare_daily_per_capita: cumulative-count differences with
    negatives clipped to 0; not smoothed, not per 100k):
        log E[cases_t] = a + r*t + day-of-week fixed effects
    over a window of GROWTH_WINDOW_DAYS (14) days starting at the growth
    start, ended at the detected peak if it would run past it. Windows
    shorter than MIN_FIT_DAYS (7) are dropped ("rise too short").
    Day-of-week effects need 8 parameters, so they are used only when the
    window has >= MIN_DOW_DAYS (12) days (>= 4 residual df); shorter windows
    are fit with the trend only (dow_effects = False).
    Dispersion = Pearson chi2 / residual df; if it exceeds
    OVERDISPERSION_LIMIT (2), SEs are quasi-Poisson (scaled by sqrt of the
    dispersion). Deviance / df is reported as a goodness-of-fit statistic.
    Dropped only if the window has < MIN_WINDOW_CASES (30) total cases or is
    too short. r <= 0 is kept (r_nonpositive flag); R_eff < 1 is allowed.
    SIR:  R_eff = 1 + r/gamma,                 SE = SE(r)/gamma
    SEIR: R_eff = (1 + r/sigma)(1 + r/gamma),  SE by the delta method
Growth rate r - comparison estimator (fit_growth_rate, earlier method)
    OLS slope of ln(smoothed daily cases per 100k) vs time from the growth
    start to the rise time-midpoint. Days whose smoothed value is below
    NEAR_ZERO_CASES_PER_DAY (in cases/day) are skipped before taking ln; at
    least MIN_FIT_DAYS retained days are required. Kept for comparison only.
SIR fit
    S, I, R as population fractions, S0 = 1 - I0, R(0) = 0, integrated from
    the wave onset with solve_ivp. The model is compared with NEW infections
    integrated over each day: C(t) = integral of beta*S*I dt (cumulative
    incidence), and model daily cases per 100k on day k is
        rho * 1e5 * [C(k+1) - C(k)]
    i.e. cumulative-incidence differences x reporting fraction x 1e5, not the
    instantaneous rate beta*S*I. Fitted: beta, log10(I0), rho; gamma fixed.
    Fit window: wave onset to wave end.
    Parameter SEs: cov = s^2 (J^T J)^-1 at the solution, s^2 = SSR / (n - p).
    The smoothed residuals are autocorrelated, so these SEs understate the
    true uncertainty.
    Final attack rate: 1 - S(inf), from the SIR final-size relation
        ln(S_inf / S0) = -R_eff * (1 - S_inf)
    (fraction of the population ever infected if the fitted wave ran to
    completion; R(0) = 0 so I0 is included).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
import statsmodels.api as sm

# Make the dashboard modules importable when scripts are run from anywhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import load_data                                    # noqa: E402
from lag_analysis import get_county_population, prepare_daily_per_capita  # noqa: E402
from wave_analysis import calculate_wave_metrics, estimate_optimal_smoothing  # noqa: E402

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")
RES_DIR = os.path.join(ANALYSIS_DIR, "results")

# ---- parameters (change here or pass explicitly) ---------------------------
INFECTIOUS_PERIOD_DAYS = 7.0          # 1/gamma
GAMMA = 1.0 / INFECTIOUS_PERIOD_DAYS
LATENT_PERIOD_DAYS = 3.0              # 1/sigma (SEIR robustness check)
SIGMA = 1.0 / LATENT_PERIOD_DAYS
WAVE_SENSITIVITY = "standard"         # dashboard detector preset for wave selection
FIRST_WAVE_CUTOFF = pd.Timestamp("2021-01-01")   # onset must be before this
MAJOR_PEAK_FRAC = 0.5                 # first wave's peak >= this x largest pre-cutoff peak
MA_WINDOW = 7                         # centered 7-day MA for all counties; None = dashboard auto-select (3/5/7/14)
NEAR_ZERO_CASES_PER_DAY = 1.0         # smoothed cases/day below this are skipped
MIN_FIT_DAYS = 7                      # min growth-window days (Poisson and OLS fits)
GROWTH_WINDOW_DAYS = 14               # Poisson GLM window length (days)
MIN_WINDOW_CASES = 30                 # Poisson GLM: drop if fewer total cases in window
MIN_DOW_DAYS = 12                     # day-of-week effects only for windows this long
OVERDISPERSION_LIMIT = 2.0            # Pearson chi2/df above this -> quasi-Poisson SEs
MIN_POPULATION = 50_000


# ---- data ------------------------------------------------------------------

def load_all():
    """Dashboard data loader: (cases_df, deaths_df, population_df), wide format."""
    return load_data()


def county_case_series(cases_df, population_df, county_name, state, fips=None,
                       ma_window=MA_WINDOW):
    """
    Smoothed daily cases per 100k for one county, built exactly as the
    dashboard's Time Lag tab builds it (auto-selected smoothing window
    unless ma_window is given).

    Returns (ts, ma_window, population) or (None, None, population) if missing.
    ts has columns Date, Daily Cases, Per100k, Per100k MA (centered MA).

    When fips is given, the cases-file county name is looked up by FIPS:
    names differ between the USAFacts cases and population files for some
    counties (e.g. "City and County of Denver" vs "Denver County"), and the
    dashboard's prepare_county_timeseries matches on name.
    """
    if fips is not None:
        match = cases_df[(cases_df["countyFIPS"] == fips) & (cases_df["State"] == state)]
        if not match.empty:
            county_name = match.iloc[0]["County Name"]
    population = get_county_population(population_df, county_name, state, fips=fips)
    if pd.isna(population):
        return None, None, population
    raw = prepare_daily_per_capita(cases_df, population_df, county_name, state,
                                   "Cases", ma_window=1, fips=fips)
    if raw.empty:
        return None, None, population
    if ma_window is None:
        ma_window = estimate_optimal_smoothing(raw["Per100k"].values)
    ts = prepare_daily_per_capita(cases_df, population_df, county_name, state,
                                  "Cases", ma_window=ma_window, fips=fips)
    return ts.reset_index(drop=True), ma_window, population


def _detect(ts, ma_window, population, sensitivity):
    return calculate_wave_metrics(
        ts["Per100k"].values, pd.DatetimeIndex(ts["Date"]),
        ma_window=ma_window, sensitivity=sensitivity,
        count_unit=100_000 / population,
    )["waves"]


def has_wave_before(ts, ma_window, population, sensitivity="conservative",
                    cutoff=FIRST_WAVE_CUTOFF):
    """True if the detector at `sensitivity` reports a wave with onset < cutoff."""
    return any(pd.Timestamp(w["start_date"]) < cutoff
               for w in _detect(ts, ma_window, population, sensitivity))


def pre_cutoff_waves(ts, ma_window, population, sensitivity=WAVE_SENSITIVITY,
                     cutoff=FIRST_WAVE_CUTOFF):
    """
    All waves at `sensitivity` with onset before `cutoff`, in onset order.
    Each is a dict with start/peak/end dates, their row indices in ts, and
    peak_height = smoothed per-100k value on the peak date.
    """
    smoothed = ts["Per100k MA"].values
    date_to_idx = {d: i for i, d in enumerate(pd.DatetimeIndex(ts["Date"]))}
    out = []
    for w in _detect(ts, ma_window, population, sensitivity):
        start = pd.Timestamp(w["start_date"])
        if start >= cutoff:
            continue
        peak, end = pd.Timestamp(w["peak_date"]), pd.Timestamp(w["end_date"])
        out.append({
            "start_date": start, "peak_date": peak, "end_date": end,
            "start_idx": date_to_idx[start], "peak_idx": date_to_idx[peak],
            "end_idx": date_to_idx[end],
            "peak_height": float(np.nan_to_num(smoothed[date_to_idx[peak]])),
        })
    return sorted(out, key=lambda x: x["start_date"])


def main_wave_index(waves, peak_frac=MAJOR_PEAK_FRAC):
    """Index of the earliest wave whose peak >= peak_frac x the largest peak."""
    if not waves:
        return None
    max_h = max(w["peak_height"] for w in waves)
    return next(i for i, w in enumerate(waves) if w["peak_height"] >= peak_frac * max_h)


def first_major_wave(ts, ma_window, population, sensitivity=WAVE_SENSITIVITY,
                     cutoff=FIRST_WAVE_CUTOFF, peak_frac=MAJOR_PEAK_FRAC):
    """
    Earliest wave at `sensitivity` with onset before `cutoff` whose peak is at
    least `peak_frac` x the largest peak among the waves with onset before
    `cutoff`. Peak height = smoothed daily cases per 100k on the detector's
    peak date (the value the dashboard charts show).

    Returns the wave dict (see pre_cutoff_waves) plus max_pre_cutoff_peak,
    n_pre_cutoff_waves and n_skipped_small (earlier pre-cutoff waves skipped
    by the peak_frac rule), or None if there is no wave with onset before
    cutoff.
    """
    waves = pre_cutoff_waves(ts, ma_window, population, sensitivity, cutoff)
    k = main_wave_index(waves, peak_frac)
    if k is None:
        return None
    return dict(waves[k],
                max_pre_cutoff_peak=max(w["peak_height"] for w in waves),
                n_pre_cutoff_waves=len(waves),
                n_skipped_small=k)


# ---- (a) growth-rate method -------------------------------------------------

def growth_window(ts, wave):
    """
    (start_idx, end_idx) inclusive. End = time midpoint of the rise; start =
    argmin of the smoothed series over [onset, end].
    """
    s, p = wave["start_idx"], wave["peak_idx"]
    e = s + (p - s) // 2
    y = ts["Per100k MA"].values[s:e + 1].astype(float)
    if np.all(~np.isfinite(y)):
        return s, e
    return s + int(np.nanargmin(y)), e


def reff_seir(r, gamma=GAMMA, sigma=SIGMA):
    return (1 + r / sigma) * (1 + r / gamma)


def fit_growth_rate(ts, wave, population, gamma=GAMMA, sigma=SIGMA,
                    near_zero_cases=NEAR_ZERO_CASES_PER_DAY,
                    min_days=MIN_FIT_DAYS):
    """
    Comparison estimator (earlier method): OLS of ln(smoothed per-100k) on
    time, growth start to rise time-midpoint. See fit_growth_poisson for the
    primary estimator.

    Returns dict: win_start, win_end, win_days, n_skipped, n_used, r, r_se,
    doubling_days, r2_log, Reff_growth, Reff_growth_se, Reff_seir,
    Reff_seir_se, ok, reason.
    """
    s, e = growth_window(ts, wave)
    y = ts["Per100k MA"].values[s:e + 1].astype(float)
    t = np.arange(len(y), dtype=float)
    floor = near_zero_cases * 100_000 / population
    mask = np.isfinite(y) & (y >= floor)
    out = {
        "win_start": ts["Date"].iloc[s], "win_end": ts["Date"].iloc[e],
        "win_days": e - s + 1, "n_skipped": int((~mask).sum()),
        "n_used": int(mask.sum()),
        "r": np.nan, "r_se": np.nan, "doubling_days": np.nan, "r2_log": np.nan,
        "Reff_growth": np.nan, "Reff_growth_se": np.nan,
        "Reff_seir": np.nan, "Reff_seir_se": np.nan,
        "ok": False, "reason": "",
    }
    if mask.sum() < min_days:
        out["reason"] = f"fewer than {min_days} usable days in growth window"
        return out
    tt, ly = t[mask], np.log(y[mask])
    X = np.column_stack([np.ones_like(tt), tt])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    resid = ly - X @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    n = len(tt)
    r = float(coef[1])
    r_se = float(np.sqrt(ss_res / (n - 2) / ((tt - tt.mean()) ** 2).sum()))
    dseir_dr = (1 + r / gamma) / sigma + (1 + r / sigma) / gamma
    out.update({
        "r": r, "r_se": r_se,
        "doubling_days": np.log(2) / r if r > 0 else np.nan,
        "r2_log": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "Reff_growth": 1 + r / gamma, "Reff_growth_se": r_se / gamma,
        "Reff_seir": reff_seir(r, gamma, sigma),
        "Reff_seir_se": abs(dseir_dr) * r_se,
        "ok": True,
    })
    return out


def reff_from_r(r, r_se, gamma=GAMMA, sigma=SIGMA):
    """(Reff_sir_formula, SE, Reff_seir_formula, SE) with delta-method SEs."""
    dseir_dr = (1 + r / gamma) / sigma + (1 + r / sigma) / gamma
    return (1 + r / gamma, r_se / gamma,
            reff_seir(r, gamma, sigma), abs(dseir_dr) * r_se)


def fit_growth_poisson(ts, wave, gamma=GAMMA, sigma=SIGMA,
                       window_days=GROWTH_WINDOW_DAYS,
                       min_cases=MIN_WINDOW_CASES,
                       min_days=MIN_FIT_DAYS,
                       min_dow_days=MIN_DOW_DAYS,
                       overdispersion_limit=OVERDISPERSION_LIMIT):
    """
    Poisson GLM (log link) of raw daily case counts on time (plus day-of-week
    fixed effects when the window has >= min_dow_days days), over up to
    `window_days` days from the growth start, ending at the peak if earlier.

    Returns dict: win_start, win_end, win_days, truncated_at_peak,
    window_cases, dow_effects, r, r_se, r_nonpositive, dispersion, quasi,
    deviance_df, Reff_sir_formula(+_se), Reff_seir_formula(+_se), ok, reason.
    """
    s, _ = growth_window(ts, wave)
    e_full = s + window_days - 1
    e = min(e_full, wave["peak_idx"])
    n = e - s + 1
    out = {"win_start": ts["Date"].iloc[s], "win_end": pd.NaT, "win_days": n,
           "truncated_at_peak": e < e_full,
           "window_cases": np.nan, "dow_effects": False,
           "r": np.nan, "r_se": np.nan, "r_nonpositive": False,
           "dispersion": np.nan, "quasi": False, "deviance_df": np.nan,
           "Reff_sir_formula": np.nan, "Reff_sir_formula_se": np.nan,
           "Reff_seir_formula": np.nan, "Reff_seir_formula_se": np.nan,
           "ok": False, "reason": ""}
    if e >= len(ts):
        out["reason"] = "window runs past end of data"
        return out
    out["win_end"] = ts["Date"].iloc[e]
    if n < min_days:
        out["reason"] = "rise too short"
        return out
    win = ts.iloc[s:e + 1]
    y = win["Daily Cases"].to_numpy(dtype=float)
    out["window_cases"] = float(y.sum())
    if y.sum() < min_cases:
        out["reason"] = f"< {min_cases} cases in window"
        return out

    t = np.arange(n, dtype=float)
    cols = [np.ones_like(t), t]
    use_dow = n >= min_dow_days
    if use_dow:
        # every weekday is present when n >= 7; drop one level as reference
        dow = pd.get_dummies(pd.DatetimeIndex(win["Date"]).dayofweek,
                             drop_first=True, dtype=float)
        cols.append(dow.to_numpy())
    X = np.column_stack(cols)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            disp = float(res.pearson_chi2 / res.df_resid)
            quasi = disp > overdispersion_limit
            if quasi:
                res = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale="X2")
    except Exception as exc:                       # noqa: BLE001
        out["reason"] = f"GLM failed: {exc}"
        return out
    if not res.converged:
        out["reason"] = "GLM did not converge"
        return out

    r, r_se = float(res.params[1]), float(res.bse[1])
    out.update({"dow_effects": use_dow, "r": r, "r_se": r_se,
                "r_nonpositive": r <= 0, "dispersion": disp, "quasi": quasi,
                "deviance_df": float(res.deviance / res.df_resid), "ok": True})
    (out["Reff_sir_formula"], out["Reff_sir_formula_se"],
     out["Reff_seir_formula"], out["Reff_seir_formula_se"]) = reff_from_r(r, r_se, gamma, sigma)
    return out


# ---- (b) SIR fit ------------------------------------------------------------

def _sir_rhs(t, z, beta, gamma):
    S, I, C = z
    inf = beta * S * I
    return [-inf, inf - gamma * I, inf]


def sir_daily_incidence(beta, I0, rho, n_days, gamma=GAMMA):
    """
    Model daily reported cases per 100k for days 0..n_days-1:
    rho * 1e5 * [C(k+1) - C(k)], C = cumulative new infections.
    """
    sol = solve_ivp(_sir_rhs, (0, n_days), [1 - I0, I0, 0.0],
                    t_eval=np.arange(n_days + 1), args=(beta, gamma),
                    method="LSODA", rtol=1e-8, atol=1e-12)
    if not sol.success or sol.y.shape[1] != n_days + 1:
        return np.full(n_days, np.nan)
    return rho * 1e5 * np.diff(sol.y[2])


def sir_final_attack_rate(R, I0):
    """1 - S_inf from ln(S_inf/S0) = -R (1 - S_inf), S0 = 1 - I0, R(0) = 0."""
    # f(S0) = R*I0 > 0 and f -> -inf as S -> 0, f concave: one root in (0, S0)
    S0 = 1 - I0
    f = lambda x: np.log(x / S0) + R * (1 - x)
    return 1 - brentq(f, 1e-300, S0, xtol=1e-14)


# Bounds: beta (1/day), log10(I0), rho
SIR_LOWER = np.array([1e-3, -10.0, 1e-3])
SIR_UPPER = np.array([5.0, -1.0, 1.0])
SIR_NAMES = ["beta", "log10_I0", "rho"]


def fit_sir(ts, wave, gamma=GAMMA):
    """
    Least-squares SIR fit of beta, I0 and reporting fraction rho to the wave's
    smoothed daily cases per 100k (onset to end), gamma fixed. Multi-start;
    the lowest-cost converged solution is kept.

    Returns dict: beta, beta_se, log10_I0, log10_I0_se, I0, rho, rho_se,
    Reff_sir, Reff_sir_se, attack_rate, r2, converged, at_bound, status_msg,
    n_days, t_dates, y_obs, y_fit.
    """
    s, e = wave["start_idx"], wave["end_idx"]
    y = ts["Per100k MA"].values[s:e + 1].astype(float)
    dates = ts["Date"].iloc[s:e + 1].reset_index(drop=True)
    good = np.isfinite(y)
    n = len(y)

    def resid(theta):
        m = sir_daily_incidence(theta[0], 10 ** theta[1], theta[2], n, gamma)
        if not np.all(np.isfinite(m)):
            return np.full(good.sum(), 1e6)
        return (m - y)[good]

    best = None
    y0 = max(np.nanmean(y[:3]), 1e-3)
    for R_init in (1.3, 1.8, 2.5, 3.5):
        for rho_init in (0.05, 0.2, 0.6):
            beta0 = R_init * gamma
            # incidence at t=0 ~ rho * 1e5 * beta * I0  ->  I0 guess
            I0_init = np.clip(y0 / (rho_init * 1e5 * beta0), 1e-9, 1e-2)
            x0 = np.clip([beta0, np.log10(I0_init), rho_init],
                         SIR_LOWER + 1e-9, SIR_UPPER - 1e-9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = least_squares(resid, x0, bounds=(SIR_LOWER, SIR_UPPER),
                                        method="trf", x_scale="jac",
                                        max_nfev=2000)
                except Exception:
                    continue
            if best is None or (res.success and (not best.success or res.cost < best.cost)):
                best = res

    out = {"n_days": n, "t_dates": dates, "y_obs": y}
    nan_keys = ["beta", "beta_se", "log10_I0", "log10_I0_se", "I0", "rho",
                "rho_se", "Reff_sir", "Reff_sir_se", "attack_rate", "r2"]
    if best is None:
        out.update({k: np.nan for k in nan_keys})
        out.update({"converged": False, "at_bound": "",
                    "status_msg": "all starts failed", "y_fit": np.full(n, np.nan)})
        return out

    beta, logI0, rho = best.x
    y_fit = sir_daily_incidence(beta, 10 ** logI0, rho, n, gamma)
    ss_res = float(np.nansum((y[good] - y_fit[good]) ** 2))
    ss_tot = float(np.nansum((y[good] - y[good].mean()) ** 2))

    # Jacobian-based covariance: s^2 (J^T J)^-1
    m_obs, p = int(good.sum()), len(best.x)
    s2 = 2 * best.cost / max(m_obs - p, 1)
    try:
        cov = s2 * np.linalg.inv(best.jac.T @ best.jac)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    span = SIR_UPPER - SIR_LOWER
    at_bound = [nm for nm, v, lo, sp in zip(SIR_NAMES, best.x, SIR_LOWER, span)
                if abs(v - lo) < 1e-4 * sp or abs(v - (lo + sp)) < 1e-4 * sp]
    I0 = float(10 ** logI0)
    Reff = float(beta / gamma)
    out.update({
        "beta": float(beta), "beta_se": float(se[0]),
        "log10_I0": float(logI0), "log10_I0_se": float(se[1]), "I0": I0,
        "rho": float(rho), "rho_se": float(se[2]),
        "Reff_sir": Reff, "Reff_sir_se": float(se[0] / gamma),
        "attack_rate": sir_final_attack_rate(Reff, I0),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "converged": bool(best.success),
        "at_bound": ";".join(at_bound),
        "status_msg": best.message,
        "y_fit": y_fit,
    })
    return out
