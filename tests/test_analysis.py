"""Deterministic tests for wave detection, lag matching, and spatial statistics."""

import numpy as np
import pandas as pd
import pytest

from lag_analysis import match_case_death_peaks
from spatial_analysis import build_adjacency_from_geojson, compute_getis_ord_gi_star
from wave_analysis import (
    calculate_wave_metrics,
    match_waves_to_national_windows,
    score_wave_significance,
)


def _two_wave_signal(n=400):
    """Synthetic daily series with two clear epidemic waves and quiet tails."""
    x = np.arange(n, dtype=float)
    wave1 = 80 * np.exp(-((x - 100) ** 2) / (2 * 15 ** 2))
    wave2 = 150 * np.exp(-((x - 280) ** 2) / (2 * 20 ** 2))
    rng = np.random.default_rng(7)
    return np.clip(wave1 + wave2 + rng.normal(0, 1.5, n), 0, None)


def test_region_detection_finds_two_waves():
    values = _two_wave_signal()
    dates = pd.date_range("2020-03-01", periods=len(values))
    metrics = calculate_wave_metrics(values, dates, ma_window=7,
                                     sensitivity="standard")
    assert metrics["number_of_waves"] == 2
    peaks = sorted(w["peak_date"] for w in metrics["waves"])
    # peaks within a week of the construction centres (days 100 and 280)
    assert abs((peaks[0] - dates[100]).days) <= 7
    assert abs((peaks[1] - dates[280]).days) <= 7
    # the larger wave carries the higher significance score
    sig = {w["peak_value"]: w["wave_significance"] for w in metrics["waves"]}
    assert sig[max(sig)] > sig[min(sig)]


def test_significance_scores_bounded():
    waves = [
        {"peak_value": 100.0, "duration_days": 60, "wave_burden": 3000.0},
        {"peak_value": 10.0, "duration_days": 20, "wave_burden": 200.0},
    ]
    scores = score_wave_significance(waves, total_burden=3200.0)
    assert len(scores) == 2 and all(0 <= s <= 100 for s in scores)
    assert scores[0] > scores[1]


def test_national_window_matching():
    waves = [
        {"wave_number": 1, "peak_date": "2020-04-10"},
        {"wave_number": 2, "peak_date": "2021-01-08"},
        {"wave_number": 3, "peak_date": "2021-09-01"},
        {"wave_number": 4, "peak_date": "2023-05-01"},
    ]
    out = match_waves_to_national_windows(waves)
    assert list(out["National Window"]) == [
        "Initial surge", "Winter 2020-21", "Delta", "Outside national windows",
    ]


def test_lag_peak_matching_greedy_chronological():
    case_peaks = [
        {"peak_date": pd.Timestamp("2020-04-01"), "peak_value": 10.0},
        {"peak_date": pd.Timestamp("2020-07-01"), "peak_value": 20.0},
    ]
    death_peaks = [
        {"peak_date": pd.Timestamp("2020-04-20"), "peak_value": 0.5},
        {"peak_date": pd.Timestamp("2020-07-25"), "peak_value": 1.0},
    ]
    matches = match_case_death_peaks(case_peaks, death_peaks, max_lag_days=90)
    assert list(matches["lag_days"]) == [19, 24]
    # a death peak can be claimed only once
    single = match_case_death_peaks(case_peaks, death_peaks[:1], max_lag_days=90)
    assert len(single) == 1 and single["lag_days"].iloc[0] == 19
    # death peaks before every case peak stay unmatched
    early = [{"peak_date": pd.Timestamp("2020-03-01"), "peak_value": 1.0}]
    assert match_case_death_peaks(case_peaks, early, max_lag_days=90).empty


def _square_ring(x0, y0):
    return [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]]


def test_adjacency_from_geojson():
    geo = {"type": "FeatureCollection", "features": [
        {"id": "00001", "geometry": {"type": "Polygon", "coordinates": _square_ring(0, 0)}},
        {"id": "00002", "geometry": {"type": "Polygon", "coordinates": _square_ring(1, 0)}},
        {"id": "00003", "geometry": {"type": "Polygon", "coordinates": _square_ring(2, 0)}},
        # corner-touching only (one shared vertex) — not a neighbour
        {"id": "00004", "geometry": {"type": "Polygon", "coordinates": _square_ring(3, 1)}},
        {"id": "00005", "geometry": {"type": "Polygon", "coordinates": _square_ring(9, 9)}},
    ]}
    adj = build_adjacency_from_geojson(geo)
    assert adj["00001"] == {"00002"}
    assert adj["00002"] == {"00001", "00003"}
    assert adj["00004"] == set()   # queen-only contact excluded
    assert adj["00005"] == set()   # island


def test_getis_ord_hotspot_detection():
    # 60 counties on a line: a high cluster at one end, a low cluster at the
    # other, mid-range values between — both extremes should be flagged.
    rng = np.random.default_rng(0)
    fips = [f"{i:05d}" for i in range(1, 61)]
    adjacency = {f: set() for f in fips}
    for i in range(59):
        adjacency[fips[i]].add(fips[i + 1])
        adjacency[fips[i + 1]].add(fips[i])

    values = np.concatenate([
        rng.normal(100, 3, 10),   # hot cluster (counties 1-10)
        rng.normal(50, 3, 40),    # background
        rng.normal(0, 3, 10),     # cold cluster (counties 51-60)
    ])
    df = pd.DataFrame({"countyFIPS": fips, "value": values})
    out = compute_getis_ord_gi_star(df, adjacency)

    hot = set(out[out["gi_category"] == "Hotspot"]["countyFIPS"])
    cold = set(out[out["gi_category"] == "Coldspot"]["countyFIPS"])
    assert hot and all(int(f) <= 11 for f in hot)
    assert cold and all(int(f) >= 50 for f in cold)
    assert out["gi_z"].notna().all()
