"""
Spatial statistics for county-level COVID data.

Provides county adjacency (derived from the bundled county GeoJSON) and
Getis-Ord Gi* hotspot detection. Both are pure data functions with no
Streamlit dependency; app.py caches their results.

Adjacency derivation: counties in the Plotly US-counties GeoJSON come from a
single topology, so counties that share a border share exact coordinate
vertices. Two counties are treated as neighbours when their boundaries share
at least two vertices (one shared vertex can be a corner point touching, as
with the Four Corners states — queen vs rook contiguity; requiring two keeps
rook-style edge sharing).

Getis-Ord Gi* (Getis & Ord 1992, 1995): for each county i, compares the
weighted sum of the metric over i's neighbourhood (including i itself) against
the global mean, normalised to a z-score. |z| > 1.96 marks statistically
significant spatial clustering at the 5% level — a hotspot (high values
surrounded by high) or coldspot (low surrounded by low).
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def build_adjacency_from_geojson(geojson: dict, min_shared_vertices: int = 2) -> Dict[str, Set[str]]:
    """
    Derive county adjacency from a US-counties GeoJSON FeatureCollection.

    Args:
        geojson:             Parsed GeoJSON dict with feature ids = 5-char FIPS.
        min_shared_vertices: Boundary vertices two counties must share to be
                             considered neighbours (2 = rook-style contiguity).

    Returns:
        Dict mapping FIPS → set of neighbouring FIPS. Counties with no
        neighbours (islands, e.g. Nantucket) map to an empty set.
    """
    vertex_owners: Dict[Tuple[float, float], Set[str]] = {}

    def _register(coords, fips):
        for ring in coords:
            for x, y in ring:
                key = (round(x, 6), round(y, 6))
                vertex_owners.setdefault(key, set()).add(fips)

    all_fips: List[str] = []
    for feat in geojson.get("features", []):
        fips = str(feat.get("id", "")).zfill(5)
        if not fips or fips == "00000":
            continue
        all_fips.append(fips)
        geom = feat.get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates", [])
        if gtype == "Polygon":
            _register(coords, fips)
        elif gtype == "MultiPolygon":
            for poly in coords:
                _register(poly, fips)

    shared_counts: Dict[Tuple[str, str], int] = {}
    for owners in vertex_owners.values():
        if len(owners) < 2:
            continue
        owners_sorted = sorted(owners)
        for i in range(len(owners_sorted)):
            for j in range(i + 1, len(owners_sorted)):
                pair = (owners_sorted[i], owners_sorted[j])
                shared_counts[pair] = shared_counts.get(pair, 0) + 1

    adjacency: Dict[str, Set[str]] = {f: set() for f in all_fips}
    for (a, b), count in shared_counts.items():
        if count >= min_shared_vertices:
            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency


def compute_getis_ord_gi_star(
    values_df: pd.DataFrame,
    adjacency: Dict[str, Set[str]],
    fips_col: str = "countyFIPS",
    value_col: str = "value",
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """
    Getis-Ord Gi* hotspot statistic with binary contiguity weights.

    Gi* includes the focal county in its own neighbourhood (the "star"
    variant). z-scores follow the standard formulation:

        Gi* = (Σ_j w_ij x_j − x̄ W_i) / (s · sqrt[(n W_i − W_i²) / (n − 1)])

    with w_ij ∈ {0,1}, W_i the neighbourhood size, x̄ and s the global mean
    and standard deviation.

    Args:
        values_df:   DataFrame with FIPS and metric columns (NaN rows dropped).
        adjacency:   Output of build_adjacency_from_geojson().
        fips_col:    FIPS column name.
        value_col:   Metric column name.
        z_threshold: |z| cutoff for hot/cold classification (1.96 ≈ p < 0.05).

    Returns:
        Copy of values_df with added columns:
            gi_z        — Gi* z-score
            gi_category — "Hotspot" | "Coldspot" | "Not significant"
            n_neighbors — neighbourhood size used (including self)
    """
    df = values_df.dropna(subset=[value_col]).copy()
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset=[fips_col]).reset_index(drop=True)

    n = len(df)
    if n < 30:
        df["gi_z"] = np.nan
        df["gi_category"] = "Not significant"
        df["n_neighbors"] = 0
        return df

    x = df[value_col].astype(float).values
    x_bar = float(x.mean())
    s = float(x.std(ddof=0))
    if s == 0:
        df["gi_z"] = 0.0
        df["gi_category"] = "Not significant"
        df["n_neighbors"] = 1
        return df

    idx_of = {f: i for i, f in enumerate(df[fips_col])}
    gi_z = np.zeros(n)
    n_nb = np.zeros(n, dtype=int)

    for i, fips in enumerate(df[fips_col]):
        neighborhood = [i] + [
            idx_of[nb] for nb in adjacency.get(fips, ()) if nb in idx_of
        ]
        w = len(neighborhood)
        n_nb[i] = w
        local_sum = float(x[neighborhood].sum())

        denom_inner = (n * w - w * w) / (n - 1)
        if denom_inner <= 0:
            gi_z[i] = 0.0
            continue
        gi_z[i] = (local_sum - x_bar * w) / (s * np.sqrt(denom_inner))

    df["gi_z"] = gi_z
    df["gi_category"] = np.select(
        [gi_z >= z_threshold, gi_z <= -z_threshold],
        ["Hotspot", "Coldspot"],
        default="Not significant",
    )
    df["n_neighbors"] = n_nb
    return df
