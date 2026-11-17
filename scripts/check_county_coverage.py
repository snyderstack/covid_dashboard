"""
County coverage audit: which county FIPS codes does each data source have?

Compares every source the dashboard joins against the USAFacts cases file
(the dashboard's county list) by 5-digit FIPS code, and lists the FIPS present
in one source but missing from another.

    python scripts/check_county_coverage.py

Sources: USAFacts cases, deaths, population; AHRF feature table; county
GeoJSON (map polygons); CDC vaccination file. Optional sources whose files are
not present (AHRF, vaccination) are reported as unavailable, not as errors.

Ignored: FIPS 00000 "Statewide Unallocated", 06000 Grand Princess Cruise Ship,
and 02270 Wade Hampton Census Area (retired code, now 02158 Kusilvak) — see
tools.NON_COUNTY_FIPS. US territories (state FIPS >= 60, e.g. Puerto Rico 72)
are not in USAFacts; they are counted on one line instead of listed.

Exit status is 1 if any county in the cases file is missing from the deaths or
population file (the joins every dashboard metric depends on), else 0.

Lives in scripts/ rather than tools/ so it doesn't collide with tools.py.
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import DATA_DIR, GEOJSON_FILENAME, NON_COUNTY_FIPS, load_data  # noqa: E402


def _is_territory(fips):
    return int(fips[:2]) >= 60


def _counties(fips_iterable):
    """Normalize to a set of 5-char FIPS, minus non-counties."""
    return {str(f).zfill(5) for f in fips_iterable} - NON_COUNTY_FIPS


def load_sources():
    """Return ({source: set of FIPS or None if unavailable}, {fips: label})."""
    cases, deaths, population = load_data()
    labels = {}
    for df in (cases, population):  # population names win (canonical)
        labels.update(zip(df["countyFIPS"], df["Location"]))

    sources = {
        "cases": _counties(cases["countyFIPS"]),
        "deaths": _counties(deaths["countyFIPS"]),
        "population": _counties(population["countyFIPS"]),
    }

    try:
        from ahrf_loader import build_ahrf_feature_table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ahrf, _ = build_ahrf_feature_table(verbose=False)
        sources["ahrf"] = _counties(ahrf["countyFIPS"]) if not ahrf.empty else None
        if not ahrf.empty and "county_name" in ahrf.columns:
            for f, name, st in zip(ahrf["countyFIPS"], ahrf["county_name"],
                                   ahrf.get("state", [""] * len(ahrf))):
                labels.setdefault(str(f).zfill(5), f"{name}, {st} (AHRF name)")
    except Exception as exc:  # missing files or parse errors
        print(f"[ahrf] unavailable: {exc}")
        sources["ahrf"] = None

    geo_path = DATA_DIR / GEOJSON_FILENAME
    if geo_path.exists():
        with open(geo_path, encoding="utf-8") as f:
            geo = json.load(f)
        sources["map_geojson"] = _counties(feat["id"] for feat in geo["features"])
        for feat in geo["features"]:
            props = feat.get("properties", {})
            labels.setdefault(str(feat["id"]).zfill(5),
                              f"{props.get('NAME', '?')} {props.get('LSAD', '')} (GeoJSON name)".strip())
    else:
        sources["map_geojson"] = None

    try:
        from vaccination_loader import load_vaccination_latest
        vax = load_vaccination_latest(str(DATA_DIR))
        sources["vaccination"] = _counties(vax["countyFIPS"]) if not vax.empty else None
    except Exception as exc:
        print(f"[vaccination] unavailable: {exc}")
        sources["vaccination"] = None

    return sources, labels


def _print_list(title, fips_set, labels):
    counties = sorted(f for f in fips_set if not _is_territory(f))
    territories = sorted(f for f in fips_set if _is_territory(f))
    if not counties and not territories:
        return
    print(f"  {title}: {len(fips_set)}")
    for f in counties:
        print(f"      {f}  {labels.get(f, '?')}")
    if territories:
        states = sorted({f[:2] for f in territories})
        print(f"      + {len(territories)} US-territory FIPS (state codes {', '.join(states)})")


def main():
    sources, labels = load_sources()
    reference = sources["cases"]

    print("\nCounty FIPS per source (non-counties excluded)")
    print("-" * 46)
    for name, fips in sources.items():
        print(f"  {name:<12} {'unavailable' if fips is None else f'{len(fips):>5,}'}")

    print("\nDifferences vs the cases file (the dashboard's county list)")
    print("-" * 60)
    for name, fips in sources.items():
        if name == "cases" or fips is None:
            continue
        missing = reference - fips
        extra = fips - reference
        print(f"\n[{name}]")
        if not missing and not extra:
            print("  identical to cases")
            continue
        _print_list(f"in cases but missing from {name}", missing, labels)
        _print_list(f"in {name} but not in cases", extra, labels)

    core_missing = (reference - sources["deaths"]) | (reference - sources["population"])
    if core_missing:
        print(f"\nFAIL: {len(core_missing)} case-file counties lack deaths or population rows")
        return 1
    print("\nOK: every county in the cases file has deaths and population rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
