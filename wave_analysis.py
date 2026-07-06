"""
Wave and outbreak analysis for COVID-19 county data.

Detects COVID waves (major outbreak events) in smoothed daily case/death data and
computes wave metrics for each county.

Detection philosophy (v3 — region-based):
    The algorithm identifies epidemiological waves rather than mathematical peaks.
    It mirrors how an epidemiologist would interpret a COVID time series:

        1. Estimate local epidemic baseline — rolling low-percentile smoothed over
           a wide window, representing inter-wave background transmission. This
           baseline adapts to each county so that post-Omicron waves are evaluated
           against local context, not the county's historical maximum.

        2. Detect epidemic regions — sustained periods where the smoothed signal
           exceeds the local baseline by a meaningful margin. Nearby elevated
           periods are merged into one region (one continuous epidemic envelope).

        3. Refine wave onset — walk back from each detected region using slope
           analysis of the smoothed signal to find where transmission first began
           rising, rather than where it crossed an arbitrary threshold.

        4. One wave per region — each epidemic region is one epidemiological event.
           The dominant peak within the region is recorded. A bimodal shape within
           a single region is one wave (one epidemic envelope); regions are only
           separated if the inter-epidemic gap exceeds the preset's merge tolerance.

    A wave significance score (0–100) combines prominence, total burden, duration,
    and burst intensity to identify the most epidemiologically important event.

Sensitivity presets:
    "conservative"  — 3–5 waves; matches widely-acknowledged national surges
    "standard"      — 4–8 waves; good default for most counties (DEFAULT)
    "sensitive"     — 6–15 waves; includes smaller regional surges

Public API:
    estimate_optimal_smoothing(daily_values)      → int
    find_waves(daily_values, ma_window, prominence, min_merge_days,
               sensitivity=None)                  → (List[Dict], Dict)
    calculate_wave_metrics(...)                   → Dict
    calculate_waves_from_values(...)              → Dict
    calculate_waves_for_county(...)               → Dict
    calculate_waves_for_all_counties(...)         → DataFrame
    score_wave_significance(waves, total_burden)  → List[float]
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks
from typing import Dict, List, Optional, Tuple


SENSITIVITY_PRESETS: Dict[str, Dict] = {
    "conservative": {
        # Adaptive baseline
        "baseline_window":        56,   # rolling window (days) for local baseline
        "baseline_percentile":    10,   # low percentile → inter-wave floor
        "baseline_smooth_window": 28,   # smoothing applied to the computed baseline

        # Epidemic region detection
        "elevation_threshold_rel": 0.50,  # signal must exceed baseline by this fraction
        "elevation_threshold_abs": 3.0,   # minimum absolute elevation above baseline
        "min_region_duration":    21,    # epidemic region must span ≥ this many days
        "region_merge_gap":       56,    # merge regions separated by ≤ this many days

        # Within-region valley splitting — separates distinct epidemic events
        # that share a continuously-elevated baseline (e.g., Delta vs Omicron).
        # Split when valley < (1 - valley_split_pct) × min(left_peak, right_peak).
        "valley_split_pct":       0.30,  # only split at very deep valleys (→ fewer waves)

        # Wave onset refinement
        "onset_lookback":         42,    # days to search backward for true onset

        # Legacy prominence-path parameters (used by the advanced-control UI
        # and the sensitivity=None detection path)
        "prominence_pct":         0.22,
        "prominence_floor_iqr_mult": 1.5,
        "min_merge_days":         45,
    },
    "standard": {
        "baseline_window":        42,
        "baseline_percentile":    10,
        "baseline_smooth_window": 21,

        "elevation_threshold_rel": 0.30,
        "elevation_threshold_abs": 2.0,
        "min_region_duration":    14,
        "region_merge_gap":       35,

        "valley_split_pct":       0.40,  # split at moderate valleys (Delta/Omicron-type)

        "onset_lookback":         28,

        "prominence_pct":         0.13,
        "prominence_floor_iqr_mult": 0.8,
        "min_merge_days":         28,
    },
    "sensitive": {
        "baseline_window":        28,
        "baseline_percentile":    15,
        "baseline_smooth_window": 14,

        "elevation_threshold_rel": 0.15,
        "elevation_threshold_abs": 1.0,
        "min_region_duration":     7,
        "region_merge_gap":       21,

        "valley_split_pct":       0.55,  # split at shallower valleys (→ more waves)

        "onset_lookback":         14,

        "prominence_pct":         0.07,
        "prominence_floor_iqr_mult": 0.4,
        "min_merge_days":         14,
    },
}

_DEFAULT_SENSITIVITY = "standard"


def estimate_optimal_smoothing(
    daily_values: np.ndarray,
    windows: List[int] = None,
) -> int:
    """
    Select a moving-average window appropriate for wave detection.

    Evaluates candidate windows using a diminishing-returns criterion: continues
    to a wider window only while marginal variance reduction exceeds 15%. For
    COVID surveillance data this reliably selects 7 days, consistent with CDC
    and WHO reporting practice for respiratory surveillance.
    """
    if windows is None:
        windows = [3, 5, 7, 14]

    clean = np.clip(np.nan_to_num(daily_values, nan=0.0), 0, None)

    if np.sum(clean > 0) < max(windows) * 2:
        return windows[0]

    prev_var: Optional[float] = None
    chosen: int = windows[0]

    for w in windows:
        smoothed = np.convolve(clean, np.ones(w) / w, mode="same")
        var = float(np.var(smoothed))

        if prev_var is not None and prev_var > 0:
            if (prev_var - var) / prev_var < 0.15:
                break

        chosen   = w
        prev_var = var

    return chosen


# Region-based epidemic detection helpers

def _estimate_epidemic_baseline(
    smoothed: np.ndarray,
    window: int = 42,
    percentile: int = 10,
    smooth_window: int = 21,
) -> np.ndarray:
    """
    Estimate local epidemic baseline via rolling low-percentile + smoothing.

    The baseline represents background (inter-wave) transmission level. A rolling
    low percentile over a wide symmetric window captures the local "floor" without
    being pulled up by epidemic peaks. The baseline is re-smoothed to avoid sharp
    transitions at wave boundaries.

    Using an adaptive local baseline rather than a global percentage of the
    historical maximum is the key improvement for detecting smaller waves: a
    post-Omicron surge that represents only 5% of the global peak but is 60%
    above recent baseline is still detected as a genuine epidemic event.
    """
    n  = len(smoothed)
    hw = window // 2
    baseline = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        baseline[i] = np.percentile(smoothed[lo:hi], percentile)

    if smooth_window > 1 and n > smooth_window:
        kernel   = np.ones(smooth_window) / smooth_window
        baseline = np.convolve(baseline, kernel, mode="same")

    return np.clip(baseline, 0.0, None)


def _detect_epidemic_regions(
    smoothed: np.ndarray,
    baseline: np.ndarray,
    threshold_rel: float = 0.30,
    threshold_abs: float = 2.0,
    min_duration: int = 14,
    merge_gap: int = 35,
) -> List[Tuple[int, int]]:
    """
    Identify sustained periods of epidemic activity above the adaptive baseline.

    A location is "in epidemic" when the smoothed signal exceeds the local baseline
    by both a relative margin and an absolute floor:

        elevation_required = baseline × threshold_rel + threshold_abs

    The two-part criterion prevents near-zero background noise from triggering
    false regions (absolute floor) while scaling to high-burden counties (relative
    component).

    Contiguous above-threshold runs are extracted, then:
      1. Nearby runs separated by ≤ merge_gap days are merged — they likely belong
         to one continuous epidemic envelope (e.g., a brief dip between the BA.1
         and BA.2 sub-waves of Omicron should not split them into two waves).
      2. Runs shorter than min_duration days are discarded as reporting artefacts
         or short-duration blips.

    Returns a list of (start_index, end_index) tuples (inclusive).
    """
    n = len(smoothed)

    elevation_required = baseline * threshold_rel + threshold_abs
    above = smoothed > (baseline + elevation_required)

    # Extract contiguous runs
    runs: List[List[int]] = []
    i = 0
    while i < n:
        if above[i]:
            j = i + 1
            while j < n and above[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1

    if not runs:
        return []

    # Merge nearby runs
    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])

    # Filter by minimum duration
    return [(s, e) for s, e in merged if (e - s + 1) >= min_duration]


def _refine_region_onset(
    smoothed: np.ndarray,
    region_start: int,
    lookback: int = 28,
) -> int:
    """
    Walk backward from the detected region start to find the true epidemic onset.

    The region threshold crossing is a lagging indicator: by the time the signal
    is clearly elevated above baseline, the outbreak has already been growing for
    days or weeks. This function identifies the onset of the rising phase.

    Method: compute first differences (day-to-day changes) of the smoothed signal
    in the lookback window. Walk backward from the region boundary to find the
    most recent point where the signal was flat or declining. The epidemic onset
    is one day after that point — the first day of the sustained upward trend.

    The lookback cap prevents the onset from being extended into a prior wave's
    resolution period.
    """
    search_start = max(0, region_start - lookback)
    if search_start >= region_start:
        return region_start

    window_slice = smoothed[search_start: region_start + 1]
    if len(window_slice) < 2:
        return region_start

    # First differences of the smoothed signal in the lookback window
    deriv = np.diff(window_slice)

    # Walk backward from end (near region_start) to find the last
    # non-positive slope (last point where signal was flat or declining).
    # Onset is one position after that point.
    onset_offset = 0  # default: walk back as far as lookback allows

    for j in range(len(deriv) - 1, -1, -1):
        if deriv[j] <= 0:
            onset_offset = j + 1
            break

    return search_start + onset_offset


def _split_regions_at_valleys(
    regions: List[Tuple[int, int]],
    smoothed: np.ndarray,
    valley_split_pct: float = 0.40,
    min_sub_duration: int = 14,
) -> List[Tuple[int, int]]:
    """
    Split epidemic regions that contain multiple distinct sub-waves.

    After region merging, some merged regions may span two genuinely distinct
    epidemic waves whose signal never fully returned to inter-wave baseline
    levels (e.g., Delta and Omicron in high-burden counties where transmission
    between these surges stayed chronically elevated). This function identifies
    and splits such regions by locating deep internal valleys.

    A valley between two sub-peaks triggers a split when:
        valley_min < (1 - valley_split_pct) × min(left_peak, right_peak)

    This is the inverse of valley-depth merging: where merging says "join if
    the valley is too shallow", splitting says "divide if the valley is deep
    enough to indicate a true inter-wave trough". The asymmetric design
    (merge gap used for proximity, valley depth used for distinctness) prevents
    bimodal intra-wave surges (e.g., BA.1/BA.2) from being over-split while
    still separating genuinely distinct waves (e.g., Delta vs Omicron).

    Sub-peaks are identified as window-maximum points within the smoothed
    signal — local maxima that dominate their surroundings over at least
    min_sub_duration // 2 days and exceed 10% of the region's global maximum.
    """
    if valley_split_pct <= 0 or not regions:
        return regions

    result: List[Tuple[int, int]] = []
    hw = max(min_sub_duration // 2, 3)

    for s, e in regions:
        region  = smoothed[s: e + 1]
        n_reg   = len(region)

        if n_reg < min_sub_duration * 2 + 1:
            result.append((s, e))
            continue

        region_max = float(np.max(region))
        sig_floor  = region_max * 0.10   # sub-peaks must exceed 10% of region max

        # Locate significant local peaks: window-maximum over ±hw days
        peak_idxs: List[int] = []
        for i in range(hw, n_reg - hw):
            lo, hi = max(0, i - hw), min(n_reg, i + hw + 1)
            if region[i] == float(np.max(region[lo:hi])) and region[i] > sig_floor:
                if not peak_idxs or (i - peak_idxs[-1]) >= hw:
                    peak_idxs.append(i)

        if len(peak_idxs) < 2:
            result.append((s, e))
            continue

        # Walk consecutive peak pairs; split at deep valleys
        sub_start = s

        for j in range(len(peak_idxs) - 1):
            pk1_loc = peak_idxs[j]
            pk2_loc = peak_idxs[j + 1]

            valley_slice   = region[pk1_loc: pk2_loc + 1]
            valley_min     = float(valley_slice.min())
            split_at_local = pk1_loc + int(np.argmin(valley_slice))
            split_at_abs   = s + split_at_local

            lower_peak = min(float(region[pk1_loc]), float(region[pk2_loc]))
            if valley_min < lower_peak * (1.0 - valley_split_pct):
                left_dur = split_at_abs - sub_start + 1
                if left_dur >= min_sub_duration:
                    result.append((sub_start, split_at_abs))
                    sub_start = split_at_abs + 1

        # Final sub-region
        final_dur = e - sub_start + 1
        if final_dur >= min_sub_duration:
            result.append((sub_start, e))
        elif result:
            result[-1] = (result[-1][0], e)   # absorb too-short tail into predecessor
        else:
            result.append((s, e))

    return result


def _merge_nearby_peaks(
    raw_waves: List[Dict],
    dates,
    min_merge_days: int,
    use_date_index: bool = True,
) -> Tuple[List[Dict], int]:
    """
    Merge wave peaks whose peak dates are within min_merge_days of each other.
    Used only by the legacy path in calculate_wave_metrics.
    """
    if not raw_waves or min_merge_days <= 0:
        return raw_waves, 0

    merged = [raw_waves[0].copy()]

    for wave in raw_waves[1:]:
        prev = merged[-1]
        if use_date_index and hasattr(dates, "iloc"):
            gap_days = (dates[wave["peak_index"]] - dates[prev["peak_index"]]).days
        else:
            gap_days = int(wave["peak_index"]) - int(prev["peak_index"])

        if gap_days <= min_merge_days:
            if wave["peak_value"] > prev["peak_value"]:
                merged[-1] = {
                    "peak_index":  wave["peak_index"],
                    "peak_value":  wave["peak_value"],
                    "start_index": prev["start_index"],
                    "end_index":   wave["end_index"],
                }
            else:
                merged[-1]["end_index"] = max(prev["end_index"], wave["end_index"])
        else:
            merged.append(wave.copy())

    return merged, len(raw_waves) - len(merged)


def score_wave_significance(
    waves: List[Dict],
    total_burden: float,
) -> List[float]:
    """
    Compute a wave significance score (0–100) for each detected wave.

    Combines four normalised dimensions:
        prominence_score  (30%): peak_value / max_peak_across_all_waves
        burden_score      (30%): wave_burden / total_series_burden
        duration_score    (20%): wave_duration / 180 days (capped at 1.0)
        intensity_score   (20%): burst intensity = (burden/duration) / max_intensity

    Burst intensity distinguishes high-intensity short surges (e.g., rapid Omicron
    peaks) from prolonged low-level elevated periods, preventing duration from
    penalising short but clearly important outbreaks.

    Higher score = more epidemiologically important.
    """
    if not waves:
        return []

    peak_values = [w.get("peak_value",    0.0) for w in waves]
    durations   = [w.get("duration_days", 1)   for w in waves]
    burdens     = [w.get("wave_burden",   0.0) for w in waves]

    max_peak  = max(peak_values) if max(peak_values) > 0 else 1.0
    total_b   = total_burden     if total_burden     > 0 else 1.0

    intensities   = [b / max(d, 1) for b, d in zip(burdens, durations)]
    max_intensity = max(intensities) if max(intensities) > 0 else 1.0

    scores = []
    for pv, dur, bur, intensity in zip(peak_values, durations, burdens, intensities):
        score = (
            (pv        / max_peak)       * 0.30 +
            (bur       / total_b)        * 0.30 +
            min(dur / 180.0, 1.0)        * 0.20 +
            (intensity / max_intensity)  * 0.20
        ) * 100.0
        scores.append(round(score, 1))

    return scores


def find_waves(
    daily_values: np.ndarray,
    ma_window: int = 7,
    prominence: float = 1000,
    min_merge_days: int = 0,
    sensitivity: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Detect epidemiologically meaningful waves in smoothed daily data.

    When sensitivity is supplied ("conservative", "standard", or "sensitive"),
    the region-based algorithm is used:
        1. Estimate adaptive local baseline (rolling low-percentile).
        2. Detect epidemic regions — sustained elevated periods above baseline.
        3. Refine each region's onset using first-derivative analysis.
        4. Assign one wave per epidemic region; dominant peak identified within.
        Merging is handled by region_merge_gap; no additional merge step is needed.

    When sensitivity is None the original prominence-based algorithm is used,
    preserving full backward compatibility for callers passing explicit values.

    Signature and return type are unchanged from prior versions.

    Returns:
        (waves, diagnostics) where each wave dict contains:
            peak_index, peak_value, start_index, end_index
    """
    diagnostics: Dict = {
        "candidate_peaks":           0,
        "after_width_filter":        0,
        "after_valley_merge":        0,
        "merged_peaks":              0,
        "final_waves":               0,
        "peak_audit_log":            [],
        # Region-based path fields
        "epidemic_regions":          0,
        "baseline_used":             False,
        "onset_refined":             False,
        "merge_applied_in_detection": False,
    }

    valid_mask = ~np.isnan(daily_values)
    filled     = daily_values.copy()
    filled[~valid_mask] = 0.0

    if valid_mask.sum() < ma_window + 2:
        return [], diagnostics

    smoothed = np.convolve(filled, np.ones(ma_window) / ma_window, mode="same")
    half     = ma_window // 2
    smoothed[:half]                 = 0.0
    smoothed[len(smoothed) - half:] = 0.0

    if smoothed.max() == 0:
        return [], diagnostics

    # Region-based epidemiological path

    if sensitivity is not None and sensitivity in SENSITIVITY_PRESETS:
        preset = SENSITIVITY_PRESETS[sensitivity]

        # Step 1: adaptive baseline
        baseline = _estimate_epidemic_baseline(
            smoothed,
            window       = preset["baseline_window"],
            percentile   = preset["baseline_percentile"],
            smooth_window= preset["baseline_smooth_window"],
        )
        diagnostics["baseline_used"] = True

        # Step 2: epidemic region detection
        raw_regions = _detect_epidemic_regions(
            smoothed, baseline,
            threshold_rel = preset["elevation_threshold_rel"],
            threshold_abs = preset["elevation_threshold_abs"],
            min_duration  = preset["min_region_duration"],
            merge_gap     = preset["region_merge_gap"],
        )

        # Step 2b: split merged regions at deep internal valleys.
        # Handles counties where transmission between distinct surges (e.g., Delta
        # and Omicron) stayed continuously elevated, preventing a clean region gap.
        raw_regions = _split_regions_at_valleys(
            raw_regions, smoothed,
            valley_split_pct  = preset["valley_split_pct"],
            min_sub_duration  = preset["min_region_duration"],
        )
        diagnostics["epidemic_regions"] = len(raw_regions)

        if not raw_regions:
            return [], diagnostics

        # Step 3: onset refinement + wave building
        waves:   List[Dict] = []
        prev_end: int       = 0
        signal_max = float(np.nanmax(smoothed)) if np.nanmax(smoothed) > 0 else 1.0

        for s, e in raw_regions:
            refined_s = _refine_region_onset(smoothed, s, preset["onset_lookback"])
            refined_s = max(refined_s, prev_end)   # no overlap with prior wave
            prev_end  = e + 1

            region_slice = smoothed[refined_s: e + 1]
            if len(region_slice) == 0:
                continue

            peak_local = int(np.argmax(region_slice))
            peak_idx   = refined_s + peak_local

            raw_val    = daily_values[peak_idx]
            peak_value = float(raw_val) if not np.isnan(raw_val) else float(smoothed[peak_idx])

            waves.append({
                "peak_index":  peak_idx,
                "peak_value":  peak_value,
                "start_index": refined_s,
                "end_index":   e,
            })

        diagnostics["onset_refined"]             = True
        diagnostics["merge_applied_in_detection"] = True
        diagnostics["candidate_peaks"]           = len(waves)
        diagnostics["after_width_filter"]        = len(waves)
        diagnostics["after_valley_merge"]        = len(waves)
        diagnostics["final_waves"]               = len(waves)

        # Audit log: one entry per wave/region in a format compatible with the UI
        audit_log = []
        for w in waves:
            idx = w["peak_index"]
            v   = float(smoothed[idx])
            audit_log.append({
                "peak_index":    idx,
                "smoothed_value": round(v, 2),
                "pct_of_max":    round(min(v / signal_max, 1.0) * 100, 1),
                "fwhm_days":     w["end_index"] - w["start_index"],
                "eff_min_width": preset["min_region_duration"],
                "width_pass":    True,
                "removed_by":    None,
            })
        diagnostics["peak_audit_log"] = audit_log

        return waves, diagnostics

    # Legacy prominence-based path (sensitivity=None)

    try:
        peak_indices, _ = _scipy_find_peaks(smoothed, prominence=prominence)
    except Exception:
        return [], diagnostics

    if len(peak_indices) == 0:
        return [], diagnostics

    diagnostics["candidate_peaks"]    = int(len(peak_indices))
    diagnostics["after_width_filter"] = diagnostics["candidate_peaks"]
    diagnostics["after_valley_merge"] = diagnostics["candidate_peaks"]

    raw_waves: List[Dict] = []
    for peak_idx in peak_indices:
        peak_value = float(
            daily_values[peak_idx] if not np.isnan(daily_values[peak_idx])
            else smoothed[peak_idx]
        )
        threshold = peak_value * 0.1

        start_idx = int(peak_idx)
        for i in range(int(peak_idx) - 1, -1, -1):
            if valid_mask[i] and daily_values[i] < threshold:
                start_idx = i
                break

        end_idx = int(peak_idx)
        for i in range(int(peak_idx) + 1, len(daily_values)):
            if valid_mask[i] and daily_values[i] < threshold:
                end_idx = i
                break

        raw_waves.append({
            "peak_index":  int(peak_idx),
            "peak_value":  peak_value,
            "start_index": start_idx,
            "end_index":   end_idx,
        })

    waves, n_merged = _merge_nearby_peaks(
        raw_waves,
        pd.RangeIndex(len(daily_values)),
        min_merge_days,
        use_date_index=False,
    )

    diagnostics["merged_peaks"] = n_merged
    diagnostics["final_waves"]  = int(len(waves))

    return waves, diagnostics


def calculate_wave_metrics(
    daily_values: np.ndarray,
    dates: pd.DatetimeIndex,
    ma_window: int = 7,
    prominence: float = 1000,
    min_merge_days: int = 0,
    sensitivity: Optional[str] = None,
) -> Dict:
    """
    Calculate wave metrics for a county's daily case or death series.

    When sensitivity is supplied ("conservative" | "standard" | "sensitive"),
    the region-based epidemiological wave detection is used (v3). Merging is
    handled internally by the region detector; no additional distance merge is
    applied post-detection.

    When sensitivity is None, legacy prominence-only detection applies (fully
    backward-compatible).

    Returns:
        Dictionary with:
            number_of_waves, waves, largest_wave, average_wave_height,
            average_wave_duration, average_time_between_waves,
            date_of_peak_wave, total_case_burden, diagnostics
        Each wave dict includes:
            wave_number, start_date, peak_date, end_date,
            peak_value, duration_days, wave_burden, wave_significance (0–100)
    """
    total_burden_val = float(np.nansum(daily_values))

    empty: Dict = {
        "number_of_waves":            0,
        "waves":                      [],
        "largest_wave":               0.0,
        "average_wave_height":        0.0,
        "average_wave_duration":      0.0,
        "average_time_between_waves": float("nan"),
        "date_of_peak_wave":          None,
        "total_case_burden":          total_burden_val,
        "diagnostics": {
            "candidate_peaks": 0, "after_width_filter": 0,
            "after_valley_merge": 0, "merged_peaks": 0, "final_waves": 0,
        },
    }

    if len(daily_values) != len(dates):
        return empty

    raw_waves, diag = find_waves(
        daily_values,
        ma_window=ma_window,
        prominence=prominence,
        min_merge_days=0,         # date-based merge handled below for legacy path
        sensitivity=sensitivity,
    )

    # Annotate audit log entries with human-readable peak dates
    for entry in diag.get("peak_audit_log", []):
        idx = entry["peak_index"]
        if 0 <= idx < len(dates):
            entry["peak_date"] = dates[idx]

    if not raw_waves:
        empty["diagnostics"] = diag
        return empty

    # Date-based distance merge — applied only for the legacy path.
    # The region-based path handles all merging internally via region_merge_gap.
    if not diag.get("merge_applied_in_detection", False):
        effective_merge = (
            SENSITIVITY_PRESETS[sensitivity]["min_merge_days"]
            if (sensitivity and sensitivity in SENSITIVITY_PRESETS)
            else min_merge_days
        )

        if effective_merge > 0:
            merged_waves: List[Dict] = [raw_waves[0].copy()]
            for wave in raw_waves[1:]:
                prev     = merged_waves[-1]
                gap_days = (dates[wave["peak_index"]] - dates[prev["peak_index"]]).days
                if gap_days <= effective_merge:
                    if wave["peak_value"] > prev["peak_value"]:
                        merged_waves[-1] = {
                            "peak_index":  wave["peak_index"],
                            "peak_value":  wave["peak_value"],
                            "start_index": prev["start_index"],
                            "end_index":   wave["end_index"],
                        }
                    else:
                        merged_waves[-1]["end_index"] = max(
                            prev["end_index"], wave["end_index"]
                        )
                else:
                    merged_waves.append(wave.copy())

            n_date_merged  = len(raw_waves) - len(merged_waves)
            raw_waves      = merged_waves
            diag["merged_peaks"] = diag.get("merged_peaks", 0) + n_date_merged
            diag["final_waves"]  = len(raw_waves)

    # Build wave detail dicts
    wave_details: List[Dict] = []
    peak_values:  List[float] = []
    durations:    List[int]   = []
    peak_dates:   List[pd.Timestamp] = []

    for idx, wave in enumerate(raw_waves, 1):
        start_date = dates[wave["start_index"]]
        peak_date  = dates[wave["peak_index"]]
        end_date   = dates[wave["end_index"]]
        duration   = max((end_date - start_date).days, 1)

        slice_vals = daily_values[wave["start_index"]: wave["end_index"] + 1]
        burden     = float(np.nansum(slice_vals))

        wave_details.append({
            "wave_number":   idx,
            "start_date":    start_date,
            "peak_date":     peak_date,
            "end_date":      end_date,
            "peak_value":    wave["peak_value"],
            "duration_days": duration,
            "wave_burden":   burden,
        })
        peak_values.append(wave["peak_value"])
        durations.append(duration)
        peak_dates.append(peak_date)

    # Significance scoring
    sig_scores = score_wave_significance(wave_details, total_burden_val)
    for wd, score in zip(wave_details, sig_scores):
        wd["wave_significance"] = score

    # "Largest wave" = highest significance score, not simply highest data point
    largest_idx = int(np.argmax(sig_scores)) if sig_scores else int(np.argmax(peak_values))

    avg_interwave = float("nan")
    if len(peak_dates) >= 2:
        gaps = [(peak_dates[i + 1] - peak_dates[i]).days
                for i in range(len(peak_dates) - 1)]
        avg_interwave = float(np.mean(gaps))

    return {
        "number_of_waves":            len(wave_details),
        "waves":                      wave_details,
        "largest_wave":               float(peak_values[largest_idx]),
        "average_wave_height":        float(np.mean(peak_values)),
        "average_wave_duration":      float(np.mean(durations)),
        "average_time_between_waves": avg_interwave,
        "date_of_peak_wave":          peak_dates[largest_idx],
        "total_case_burden":          total_burden_val,
        "diagnostics":                diag,
    }


def calculate_waves_from_values(
    daily_values: np.ndarray,
    dates: pd.DatetimeIndex,
    ma_window: int = 7,
    prominence: float = 1000,
    min_merge_days: int = 0,
    sensitivity: Optional[str] = None,
) -> Dict:
    """
    Run wave analysis on a pre-prepared values array.

    Preferred entry point when the caller has already normalised the data
    (per-capita division, etc.). Delegates to calculate_wave_metrics().
    """
    return calculate_wave_metrics(
        daily_values, dates,
        ma_window=ma_window,
        prominence=prominence,
        min_merge_days=min_merge_days,
        sensitivity=sensitivity,
    )


def calculate_waves_for_county(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    daily_cases_df: pd.DataFrame,
    daily_deaths_df: pd.DataFrame,
    county_name: str,
    state: str,
    ma_window: int = 7,
    prominence: float = 1000,
    min_merge_days: int = 0,
    sensitivity: Optional[str] = None,
) -> Dict:
    """
    Calculate wave metrics for a specific county.

    Accepts an optional sensitivity preset; when supplied, the preset
    parameters govern the region-based detection.
    """
    cases_row        = cases_df[(cases_df["County Name"] == county_name) & (cases_df["State"] == state)]
    daily_cases_row  = daily_cases_df[(daily_cases_df["County Name"] == county_name) & (daily_cases_df["State"] == state)]
    daily_deaths_row = daily_deaths_df[(daily_deaths_df["County Name"] == county_name) & (daily_deaths_df["State"] == state)]

    if cases_row.empty or daily_cases_row.empty:
        return {"error": "County not found"}

    identifier_cols = ["countyFIPS", "County Name", "State", "StateFIPS", "Location"]
    date_cols = [col for col in daily_cases_df.columns if col not in identifier_cols]
    dates     = pd.to_datetime(date_cols)

    daily_cases = pd.to_numeric(
        daily_cases_row.iloc[0, daily_cases_row.columns.get_loc(date_cols[0]):],
        errors="coerce",
    ).values

    daily_deaths = pd.to_numeric(
        daily_deaths_row.iloc[0, daily_deaths_row.columns.get_loc(date_cols[0]):],
        errors="coerce",
    ).values if not daily_deaths_row.empty else np.zeros(len(daily_cases))

    cases_metrics  = calculate_wave_metrics(
        daily_cases,  dates, ma_window, prominence, min_merge_days, sensitivity
    )
    deaths_metrics = calculate_wave_metrics(
        daily_deaths, dates, ma_window, prominence, min_merge_days, sensitivity
    )

    return {
        "county_name": county_name,
        "state":       state,
        "cases":       cases_metrics,
        "deaths":      deaths_metrics,
    }


def calculate_waves_for_all_counties(
    cases_df: pd.DataFrame,
    deaths_df: pd.DataFrame,
    daily_cases_df: pd.DataFrame,
    daily_deaths_df: pd.DataFrame,
    ma_window: int = 7,
    prominence: float = 1000,
    min_merge_days: int = 0,
    sensitivity: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate wave metrics for all counties."""
    results = []

    for _, row in cases_df.iterrows():
        county_name = row.get("County Name", "Unknown")
        state       = row.get("State", "Unknown")

        metrics = calculate_waves_for_county(
            cases_df, deaths_df, daily_cases_df, daily_deaths_df,
            county_name, state, ma_window, prominence, min_merge_days, sensitivity,
        )

        if "error" not in metrics:
            results.append({
                "countyFIPS":               row.get("countyFIPS"),
                "County Name":              county_name,
                "State":                    state,
                "case_waves":               metrics["cases"]["number_of_waves"],
                "case_largest_wave":        metrics["cases"]["largest_wave"],
                "case_avg_wave_height":     metrics["cases"]["average_wave_height"],
                "case_avg_wave_duration":   metrics["cases"]["average_wave_duration"],
                "case_avg_interwave_days":  metrics["cases"]["average_time_between_waves"],
                "case_peak_wave_date":      metrics["cases"]["date_of_peak_wave"],
                "case_total_burden":        metrics["cases"]["total_case_burden"],
                "death_waves":              metrics["deaths"]["number_of_waves"],
                "death_largest_wave":       metrics["deaths"]["largest_wave"],
                "death_avg_wave_height":    metrics["deaths"]["average_wave_height"],
                "death_avg_wave_duration":  metrics["deaths"]["average_wave_duration"],
                "death_avg_interwave_days": metrics["deaths"]["average_time_between_waves"],
                "death_peak_wave_date":     metrics["deaths"]["date_of_peak_wave"],
                "death_total_burden":       metrics["deaths"]["total_case_burden"],
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from tools import load_data, precompute_daily_diffs

    print("Loading data...")
    cases, deaths, pop = load_data()
    daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)

    test_counties = [
        ("Los Angeles County", "CA"),
        ("New York County",    "NY"),
        ("Cook County",        "IL"),
        ("Loving County",      "TX"),
        ("Hamilton County",    "KS"),
    ]

    for county, state in test_counties:
        print(f"\n{county}, {state}")
        for sens in ["conservative", "standard", "sensitive"]:
            metrics = calculate_waves_for_county(
                cases, deaths, daily_cases, daily_deaths,
                county, state, ma_window=7, sensitivity=sens,
            )
            if "error" in metrics:
                print(f"  [{sens:12s}] NOT FOUND")
                continue
            m = metrics["cases"]
            print(
                f"  [{sens:12s}] {m['number_of_waves']} waves | "
                f"largest={m['largest_wave']:.0f} | "
                f"avg_dur={m['average_wave_duration']:.0f}d"
            )
