"""County coverage: every county is listed and every lookup is keyed by FIPS.

The USAFacts cases/deaths files and the population file name 57 counties
differently (e.g. "City and County of Honolulu" vs "Honolulu County"), so any
join on county name silently drops them. The synthetic test runs everywhere;
the real-data tests are skipped when data/ is not present.

Also covers the county GeoJSON re-keying of retired FIPS (Kusilvak 02158,
Oglala Lakota 46102) and the display-name corrections.
"""

import json

import pandas as pd
import pytest

from tests.conftest import DATES, _wide
from lag_analysis import analyze_county_lag, get_county_population
from tools import (
    DATA_DIR,
    GEOJSON_FILENAME,
    NON_COUNTY_FIPS,
    RETIRED_FIPS,
    apply_canonical_county_names,
    build_county_index,
    canonical_county_names,
    county_mask,
    find_data_corrections,
    get_population_column,
    load_county_geojson,
    load_data,
    precompute_daily_diffs,
    prepare_dashboard_data,
    precompute_per_capita,
    prepare_choropleth_for_date,
    prepare_county_timeseries,
    remap_retired_geojson_fips,
    rolling_cfr,
)
from wave_analysis import calculate_waves_for_county

HAVE_DATA = all(
    (DATA_DIR / f).exists()
    for f in ("covid_confirmed_usafacts.csv", "covid_deaths_usafacts.csv",
              "covid_county_population_usafacts.csv")
)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="USAFacts CSVs not in data/")

# FIPS -> canonical label the pickers should show.
SPOT_CHECK = {
    "15003": "Honolulu County, HI",
    "06075": "San Francisco County, CA",
    "11001": "Washington, DC",
    "51810": "Virginia Beach city, VA",
}


def test_name_mismatch_resolved_by_fips():
    """A county named differently in cases vs population is still found."""
    cases = _wide([("15003", "City and County of Honolulu", "HI", list(range(10)))])
    pop = pd.DataFrame({"countyFIPS": ["15003"], "County Name": ["Honolulu County"],
                        "State": ["HI"], "population": [1_000_000]})

    idx = build_county_index(cases, pop)
    assert idx["Location"].tolist() == ["Honolulu County, HI"]

    canon = apply_canonical_county_names(cases, pop)
    assert canon.loc[0, "Location"] == "Honolulu County, HI"
    assert county_mask(pop, fips="15003").sum() == 1
    assert county_mask(pop, fips=15003).sum() == 1  # int FIPS is padded
    ts = prepare_county_timeseries(cases, "ignored", "HI", fips="15003")
    assert len(ts) == len(DATES)
    assert get_county_population(pop, "City and County of Honolulu", "HI",
                                 fips="15003") == 1_000_000


def test_geojson_retired_fips_are_rekeyed():
    geo = {"features": [{"id": "02270"}, {"id": "46113"}, {"id": "01001"}]}
    ids = [f["id"] for f in remap_retired_geojson_fips(geo)["features"]]
    assert ids == ["02158", "46102", "01001"]
    # Never creates a duplicate when the current code already has a polygon.
    geo = {"features": [{"id": "02270"}, {"id": "02158"}]}
    ids = [f["id"] for f in remap_retired_geojson_fips(geo)["features"]]
    assert ids == ["02270", "02158"]


def test_display_name_corrections():
    pop = pd.DataFrame({
        "countyFIPS": ["51115", "51710", "51520", "51019", "24510", "02270"],
        "County Name": ["Matthews County", "Norfolk City", "Bristol city",
                        "Bedford County", "Baltimore City", "Wade Hampton Census Area"],
        "State": ["VA", "VA", "VA", "VA", "MD", "AK"],
        "population": [1, 1, 1, 1, 1, 0],
    })
    names = canonical_county_names(pop)
    assert names["51115"] == "Mathews County"
    assert names["51710"] == "Norfolk city"
    assert names["51520"] == "Bristol city"
    assert names["51019"] == "Bedford County"   # VA county, not a city
    assert names["24510"] == "Baltimore City"   # only Virginia is restyled
    assert "02270" not in names                 # retired code


def test_retired_fips_dropped_from_dashboard_population():
    cases = _wide([("02158", "Kusilvak Census Area", "AK", list(range(10)))])
    pop = pd.DataFrame({"countyFIPS": ["02158", "02270"],
                        "County Name": ["Kusilvak Census Area", "Wade Hampton Census Area"],
                        "State": ["AK", "AK"], "population": [8314, 0]})
    pop["Location"] = pop["County Name"] + ", " + pop["State"]
    _, _, pop_out, _ = prepare_dashboard_data(cases, cases.copy(), pop)
    assert pop_out["countyFIPS"].tolist() == ["02158"]


@pytest.fixture(scope="module")
def real_data():
    cases, deaths, pop, _ = prepare_dashboard_data(*load_data())
    index = build_county_index(cases)
    return cases, deaths, pop, index


@needs_data
def test_every_cases_county_is_in_dashboard_list(real_data):
    cases, _, _, index = real_data
    raw_cases, _, _ = load_data()
    expected = set(raw_cases["countyFIPS"]) - NON_COUNTY_FIPS
    listed = set(index["countyFIPS"])
    missing = sorted(expected - listed)
    assert not missing, f"{len(missing)} counties missing from county list: {missing[:20]}"
    assert index["Location"].is_unique
    assert index["countyFIPS"].is_unique


@needs_data
def test_every_cases_county_has_population(real_data):
    cases, _, pop, index = real_data
    pop_col = get_population_column(pop)
    valid_pop = pop[pd.to_numeric(pop[pop_col], errors="coerce") > 0]
    missing = sorted(set(index["countyFIPS"]) - set(valid_pop["countyFIPS"]))
    assert not missing, f"counties with no population row: {missing}"


@needs_data
@pytest.mark.parametrize("fips,label", SPOT_CHECK.items())
def test_spot_check_counties_load_with_data(real_data, fips, label):
    cases, deaths, pop, index = real_data
    location_fips = dict(zip(index["Location"], index["countyFIPS"]))
    assert location_fips.get(label) == fips, f"{label} not in county list"

    state = label.rsplit(", ", 1)[1]
    name = label.rsplit(", ", 1)[0]

    ts = prepare_county_timeseries(cases, name, state, fips=fips)
    assert not ts.empty and ts["Cases"].iloc[-1] > 0

    population = get_county_population(pop, name, state, fips=fips)
    assert population > 0

    assert county_mask(deaths, fips=fips).sum() == 1
    assert isinstance(find_data_corrections(cases, name, state, fips=fips), pd.DataFrame)
    assert not rolling_cfr(cases, deaths, name, state, fips=fips).empty

    daily_cases, daily_deaths = precompute_daily_diffs(cases, deaths)
    waves = calculate_waves_for_county(cases, deaths, daily_cases, daily_deaths,
                                       name, state, sensitivity="standard", fips=fips)
    assert "error" not in waves

    lag = analyze_county_lag(cases, deaths, pop, name, state, fips=fips)
    assert "error" not in lag

    pc_cases, _ = precompute_per_capita(cases, deaths, pop)
    last_date = [c for c in cases.columns if c[:2] == "20"][-1]
    choro = prepare_choropleth_for_date(pc_cases, last_date, cases, deaths, pop)
    row = choro[choro["countyFIPS"] == fips]
    assert len(row) == 1
    assert row["Location"].iloc[0] == label
    assert row["population"].iloc[0] > 0 and row["value"].iloc[0] > 0


@needs_data
def test_dashboard_population_has_no_retired_fips(real_data):
    _, _, pop, _ = real_data
    assert not set(pop["countyFIPS"]) & set(RETIRED_FIPS)


@needs_data
@pytest.mark.skipif(not (DATA_DIR / GEOJSON_FILENAME).exists(),
                    reason="county GeoJSON not in data/")
def test_map_draws_every_county(real_data):
    _, _, _, index = real_data
    geo = load_county_geojson()
    geo_ids = {f["id"] for f in geo["features"]}
    missing = sorted(set(index["countyFIPS"]) - geo_ids)
    assert not missing, f"counties with no map polygon: {missing}"
    assert {"02158", "46102"} <= geo_ids
    assert not geo_ids & set(RETIRED_FIPS)
    # The raw file on disk is untouched.
    with open(DATA_DIR / GEOJSON_FILENAME, encoding="utf-8") as f:
        assert "02270" in {feat["id"] for feat in json.load(f)["features"]}
