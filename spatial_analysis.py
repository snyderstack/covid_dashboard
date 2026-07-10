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

from typing import Dict, List, Set, Tuple

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


def compute_morans_i(
    values_df: pd.DataFrame,
    adjacency: Dict[str, Set[str]],
    fips_col: str = "countyFIPS",
    value_col: str = "value",
) -> dict:
    """
    Global Moran's I spatial autocorrelation with binary contiguity weights.

    Moran's I answers "do similar values cluster geographically?" for the map
    as a whole (Gi* answers it locally, county by county). I ranges roughly
    from −1 (checkerboard dispersion) through E[I] = −1/(n−1) ≈ 0 (spatial
    randomness) to +1 (strong clustering). Significance uses the analytic
    z-score under the normality assumption (Cliff & Ord 1981):

        I = (n / S0) · Σᵢⱼ wᵢⱼ zᵢ zⱼ / Σᵢ zᵢ²

    Returns:
        Dict with keys: I, expected, z, p_value, n, n_edges.
        NaN statistics when fewer than 30 counties or no edges exist.
    """
    df = values_df.dropna(subset=[value_col]).copy()
    df[fips_col] = df[fips_col].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset=[fips_col]).reset_index(drop=True)

    n = len(df)
    empty = {"I": np.nan, "expected": np.nan, "z": np.nan,
             "p_value": np.nan, "n": n, "n_edges": 0}
    if n < 30:
        return empty

    idx_of = {f: i for i, f in enumerate(df[fips_col])}
    x = df[value_col].astype(float).values
    z = x - x.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return empty

    cross = 0.0
    degrees = np.zeros(n)
    n_edges = 0
    for fips, nbrs in adjacency.items():
        i = idx_of.get(fips)
        if i is None:
            continue
        for nb in nbrs:
            j = idx_of.get(nb)
            if j is None:
                continue
            cross += z[i] * z[j]      # ordered pairs — each edge counted twice
            degrees[i] += 1
            n_edges += 1

    if n_edges == 0:
        return empty

    s0 = float(n_edges)               # Σ wij over ordered pairs
    I = (n / s0) * (cross / denom)

    # Analytic moments under normality (binary symmetric weights):
    # S1 = ½ Σ (wij + wji)² = 2·S0 ;  S2 = Σᵢ (row_sumᵢ + col_sumᵢ)² = 4 Σ degᵢ²
    e_i = -1.0 / (n - 1)
    s1 = 2.0 * s0
    s2 = 4.0 * float((degrees ** 2).sum())
    var_i = (
        (n * n * s1 - n * s2 + 3.0 * s0 * s0)
        / ((n * n - 1.0) * s0 * s0)
        - e_i * e_i
    )
    if var_i <= 0:
        return {**empty, "I": I, "expected": e_i}

    z_score = (I - e_i) / np.sqrt(var_i)
    # two-sided normal p-value via erfc
    from math import erfc, sqrt
    p = erfc(abs(z_score) / sqrt(2.0))

    return {"I": float(I), "expected": float(e_i), "z": float(z_score),
            "p_value": float(p), "n": n, "n_edges": n_edges // 2}


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
