"""Shared synthetic fixtures for the dashboard test suite.

All tests use small constructed datasets so results are deterministic and the
suite runs in seconds without the real data files.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATES = [f"2020-01-{d:02d}" for d in range(22, 32)]  # 10 daily columns


def _wide(rows):
    """Build a wide-format USAFacts-style dataframe from (fips, name, state, values)."""
    records = []
    for fips, name, state, values in rows:
        rec = {"countyFIPS": fips, "County Name": name, "State": state,
               "StateFIPS": fips[:2], "Location": f"{name}, {state}"}
        rec.update(dict(zip(DATES, values)))
        records.append(rec)
    return pd.DataFrame(records)


@pytest.fixture
def cases_df():
    return _wide([
        ("01001", "Alpha County", "AA", [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]),
        ("01003", "Beta County",  "AA", [0, 2, 4, 8, 16, 20, 24, 28, 30, 40]),
        # cumulative dip at index 5 → one negative daily diff (data correction)
        ("02001", "Gamma County", "BB", [0, 5, 10, 20, 30, 25, 40, 50, 60, 70]),
        ("00000", "Statewide Unallocated", "AA", [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]),
    ])


@pytest.fixture
def deaths_df():
    return _wide([
        ("01001", "Alpha County", "AA", [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]),
        ("01003", "Beta County",  "AA", [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        ("02001", "Gamma County", "BB", [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
        ("00000", "Statewide Unallocated", "AA", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
    ])


@pytest.fixture
def population_df():
    return pd.DataFrame({
        "countyFIPS":  ["01001", "01003", "02001", "00000"],
        "County Name": ["Alpha County", "Beta County", "Gamma County",
                        "Statewide Unallocated"],
        "State":       ["AA", "AA", "BB", "AA"],
        "population":  [10_000, 20_000, 50_000, 0],
    })
