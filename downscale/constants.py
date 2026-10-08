from __future__ import annotations

import numpy as np

#: Standard gravity, used to convert geopotential to height [m s^-2].
G = 9.80665

#: Standard atmosphere temperature lapse rate [K m^-1].
STD_LAPSE = 6.5e-3

#: Named bounding boxes, ``(lat_min, lat_max, lon_min, lon_max)``, lon in -180..180.
REGION_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "europe": (35.0, 72.0, -25.0, 45.0),
    "north-atlantic": (20.0, 70.0, -80.0, 10.0),
    "conus": (24.0, 50.0, -125.0, -66.0),
    "france": (40.0, 53.0, -7.0, 11.0),
}

#: Sub-regions used to stratify station scores, ``((lat_min, lat_max), (lon_min, lon_max))``.
SCORING_REGIONS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "france": ((41.0, 51.5), (-5.5, 10.0)),
    "alps": ((44.0, 46.8), (5.8, 7.8)),
    "pyrenees": ((42.2, 43.3), (-1.8, 3.0)),
    "massif-central": ((44.5, 46.0), (2.0, 4.0)),
    "mediterranean": ((42.5, 44.0), (3.0, 7.5)),
    "atlantic": ((43.5, 48.5), (-5.0, -0.5)),
    "north": ((48.5, 51.0), (-4.5, 8.0)),
}

#: Canonical station variables: unit and the plausible range used as a gross-error filter.
STATION_VARIABLES: dict[str, dict] = {
    "t2m": {"unit": "K", "range": (223.15, 333.15)},
    "tx1h": {"unit": "K", "range": (223.15, 333.15)},
    "tn1h": {"unit": "K", "range": (223.15, 333.15)},
}

#: Plausible range for gridded 2 m temperature [K], used in download checks.
T2M_RANGE_K = (220.0, 330.0)

#: Dimension holding ensemble members; any dimension whose name starts with this counts.
ENSEMBLE_DIM = "ensemble"

#: Metric names that require an ensemble forecast.
ENSEMBLE_METRICS = frozenset({"crps", "ens_mean_rmse", "spread", "ssr"})

#: Static (time-invariant) model input channels, in input order.
STATIC_CHANNELS = ["cerra_orography", "orography_diff", "land_sea_mask", "lat", "lon"]

#: Time-varying calendar channels, in input order.
TIME_CHANNELS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

#: Full model input channel order, excluding noise channels.
CHANNELS = ["era5_t2m", *STATIC_CHANNELS, *TIME_CHANNELS]

#: Copernicus Climate Data Store dataset identifier for the high-resolution target.
CERRA_DATASET = "reanalysis-cerra-single-levels"

#: Margin added around the target grid when cropping the coarse source [degrees].
COARSE_CROP_MARGIN_DEG = 1.5

#: Maximum distance from a station to the nearest fine grid cell [km]; beyond this the value is NaN.
MAX_STATION_DISTANCE_KM = 15.0

#: Fraction of fine grid points allowed to fall outside the coarse forecast domain.
MAX_FRACTION_OUTSIDE = 0.02

#: Version tag written into published data packs.
PACK_VERSION = "0.1"

#: Earth radius used for great-circle distances [km].
EARTH_RADIUS_KM = 6371.0


def elevation_edges(edges: list[float] | None = None) -> np.ndarray:
    """Return elevation band edges as an array.

    :param edges: Band edges in metres; ``None`` yields the default bands.
    :type edges: list[float] | None
    :returns: Monotonic array of band edges.
    :rtype: numpy.ndarray
    """
    return np.asarray(edges if edges is not None else [0, 200, 500, 1000, 1500, 2500, 5000], dtype=float)