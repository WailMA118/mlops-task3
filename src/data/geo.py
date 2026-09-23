"""
Geolocation helpers: zip-prefix -> coordinates lookup and haversine distance.

Refactor of Notebook 1's "STEP 0" cell (zip_geo aggregation + haversine_km).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371


def build_zip_geo_lookup(geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw geolocation table (many points per zip prefix) into one
    row per zip prefix using mean lat/lng. Must run before any join, exactly
    as in Notebook 1: aggregate the one-to-many side first.
    """
    return (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
    )


def haversine_km(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Great-circle distance between two coordinates, in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))