"""Deterministic tests for modeling helpers and the similarity finder."""

import numpy as np
import pandas as pd
import pytest

from county_features import (
    compute_bivariate_correlation,
    compute_ols_trend,
    find_similar_counties,
    generate_county_insights,
)
from modeling import _kmeans_numpy, _ols_fit, compute_county_clusters, compute_vif


def _synthetic_master(n=200, seed=3):
    """County table with two structural blocks and a linear outcome."""
    rng = np.random.default_rng(seed)
    metro = rng.random(n) < 0.5
    density = np.where(metro, rng.normal(2000, 300, n), rng.normal(30, 10, n))
    income = np.where(metro, rng.normal(90_000, 8_000, n), rng.normal(50_000, 6_000, n))
    return pd.DataFrame({
        "countyFIPS": [f"{i:05d}" for i in range(1, n + 1)],
        "County Name": [f"County {i}" for i in range(1, n + 1)],
        "State": ["XX"] * n,
        "population": np.where(metro, rng.normal(5e5, 5e4, n), rng.normal(2e4, 5e3, n)),
        "pop_density_per_sqmi": density,
        "median_family_income": income,
        "pct_pop_65plus": rng.normal(18, 3, n),
        "pcp_per_100k": rng.normal(60, 15, n),
        "pct_college_4yr": rng.normal(25, 6, n),
        "unemployment_rate": rng.normal(5, 1, n),
        "rucc_code": np.where(metro, 1.0, 8.0),
        "deaths_per_100k": 400 - income / 1000 + rng.normal(0, 5, n),
        "cases_per_100k": rng.normal(25_000, 3_000, n),
    })


def test_find_similar_counties_prefers_same_block():
    df = _synthetic_master()
    metro_fips = df[df["rucc_code"] == 1.0]["countyFIPS"].iloc[0]
    peers = find_similar_counties(df, metro_fips, n=10)
    assert len(peers) == 10
    assert metro_fips not in set(peers["countyFIPS"])
    # a metro county's nearest structural peers should all be metro
    assert (peers["rucc_code"] == 1.0).all()
    assert peers["similarity_distance"].is_monotonic_increasing


def test_find_similar_counties_handles_missing():
    df = _synthetic_master()
    df.loc[0, "median_family_income"] = np.nan   # reference county incomplete
    assert find_similar_counties(df, df["countyFIPS"].iloc[0], n=10).empty


def test_same_column_correlation_does_not_crash():
    # Regression test: County Factors lets vaccination appear as both factor
    # and outcome; selecting the same column on both axes (or a vaccination
    # outcome triggering the rankings loop) previously raised
    # "TypeError: arg must be a list, tuple, 1-d array, or Series" because
    # df[[c, c]] produces duplicate columns whose access yields a DataFrame.
    df = _synthetic_master()
    res = compute_bivariate_correlation(df, "median_family_income", "median_family_income")
    assert res["pearson_r"] == 1.0 and res["r_squared"] == 1.0
    assert res["n"] > 0 and res["x_std"] == res["y_std"]
    ols = compute_ols_trend(df, "median_family_income", "median_family_income")
    assert ols["slope"] == pytest.approx(1.0) and ols["intercept"] == pytest.approx(0.0, abs=1e-6)


def test_insight_engine_flags_low_vaccination():
    rng = np.random.default_rng(5)
    df = _synthetic_master(seed=5)
    df["vax_complete_pct"] = rng.normal(55, 6, len(df))
    df["deaths_per_100k"] = 500 - 4 * df["vax_complete_pct"] + rng.normal(0, 10, len(df))
    # target county: structurally typical, but far below peers on vaccination
    df.loc[0, "vax_complete_pct"] = 30.0
    df.loc[0, "deaths_per_100k"] = 500 - 4 * 30.0

    peers = find_similar_counties(df, df["countyFIPS"].iloc[0], n=10)
    ins = generate_county_insights(df, df["countyFIPS"].iloc[0], peers)

    assert ins is not None and ins["n_peers"] >= 5
    assert ins["peer_percentile"] > 75          # worse than most peers
    risk_labels = [f["label"] for f in ins["risk_factors"]]
    assert "vaccination coverage" in risk_labels
    vax = next(f for f in ins["risk_factors"] if f["label"] == "vaccination coverage")
    assert vax["county_value"] == pytest.approx(30.0)
    assert vax["assoc_r"] < 0                   # empirically negative association
    # graceful degradation
    assert generate_county_insights(df, "99999", peers) is None
    assert generate_county_insights(df, df["countyFIPS"].iloc[0], peers.iloc[0:0]) is None


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    df = pd.DataFrame({
        "median_family_income": a,
        "per_capita_income": a * 0.98 + rng.normal(0, 0.05, n),  # near-duplicate
        "unemployment_rate": rng.normal(size=n),                 # independent
    })
    vif = compute_vif(df, list(df.columns))
    vals = dict(zip(vif["Variable"], vif["VIF"]))
    assert vals["Unemployment Rate (%)"] < 1.5
    assert vals["Median Family Income ($)"] > 10
    assert vals["Per Capita Income ($)"] > 10


def test_ols_hc3_close_to_classical_when_homoscedastic():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(0, 1, n)     # constant variance
    X = np.column_stack([np.ones(n), x])
    res = _ols_fit(X, y)
    assert res["beta"][1] == pytest.approx(3.0, abs=0.15)
    # under homoscedasticity, HC3 and classical SEs agree within ~15%
    assert res["se_hc3"][1] == pytest.approx(res["se"][1], rel=0.15)
    assert np.all(np.isfinite(res["p_vals_hc3"][:2]))


def test_kmeans_numpy_separates_blocks():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (60, 2))])
    labels = _kmeans_numpy(X, k=2)
    # each true block maps to a single cluster label
    assert len(set(labels[:50])) == 1
    assert len(set(labels[50:])) == 1
    assert labels[0] != labels[-1]


def test_compute_county_clusters_end_to_end():
    df = _synthetic_master()
    features = ["population", "pop_density_per_sqmi", "median_family_income",
                "pct_pop_65plus", "rucc_code"]
    assign, profile, err = compute_county_clusters(df, features, k=2)
    assert err is None
    assert set(assign["cluster"]) == {0, 1}
    # clusters recover the metro/nonmetro split (allowing a couple of strays)
    merged = assign.merge(df[["countyFIPS", "rucc_code"]], on="countyFIPS")
    purity = (
        merged.groupby("cluster")["rucc_code"]
        .agg(lambda s: max((s == 1.0).mean(), (s == 8.0).mean()))
    )
    assert (purity > 0.95).all()
    assert profile["counties"].sum() == len(df)
