"""
Gridded forecast -> station points, and the baselines built on it.

The ladder this module provides (each rung must beat the one below it for
downscaling to have earned its place):

    nearest          value of the nearest grid cell
    bilinear         4-neighbour interpolation in lat/lon
    bilinear_lapse   bilinear, then corrected for the height difference
                     between the station and the model's own orography

and the diagnostic that tells you how much of the error is the FORECAST and
how much is the grid-vs-point mismatch:

    analysis_baseline   the same interpolation applied to the ERA5 ANALYSIS
                        at the valid time. It contains no forecast error at
                        all, so whatever error remains against stations is
                        representativeness error -- the floor that no
                        downscaling of a forecast can be expected to beat.

Conventions:
  * grid longitudes may be 0..360 and non-monotonic across the prime
    meridian (as the cropped France zarrs are). They are converted to signed
    longitudes and sorted internally; stations use signed longitudes.
  * no periodic wrap: a station outside the grid box gets NaN. Fine for a
    regional crop, wrong for a global grid with stations near +-180.
  * temperatures in K, heights in m, lapse rate in K/m.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .regions import to_signed_lon

logger = logging.getLogger(__name__)

G = 9.80665                 # standard gravity, geopotential -> height
STD_LAPSE = 6.5e-3          # K/m, the ICAO standard atmosphere


# ---------------------------------------------------------------------------
# core interpolator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridInterpolator:
    """Weights from a regular lat/lon grid to a set of points, built once and
    applied to any array that has that grid on its (lat, lon) axes.

    Building is the only place that looks at coordinates, so applying it to
    ten members x forty lead times is a handful of vectorised gathers."""

    lat_order: np.ndarray    # indices that put the grid's lat axis ascending
    lon_order: np.ndarray    # indices that put the grid's signed-lon axis ascending
    i0: np.ndarray
    j0: np.ndarray
    wy: np.ndarray           # weight of the northern neighbour, 0..1
    wx: np.ndarray           # weight of the eastern neighbour, 0..1
    inside: np.ndarray       # False -> point lies outside the grid -> NaN
    method: str
    n_lat: int
    n_lon: int

    @classmethod
    def build(cls, grid_lat, grid_lon, point_lat, point_lon, method: str = "bilinear"):
        if method not in ("bilinear", "nearest"):
            raise ValueError("method must be 'bilinear' or 'nearest'")
        grid_lat = np.asarray(grid_lat, dtype="f8")
        glon = to_signed_lon(grid_lon)
        lat_order = np.argsort(grid_lat, kind="stable")
        lon_order = np.argsort(glon, kind="stable")
        glat, glon = grid_lat[lat_order], glon[lon_order]
        if glat.size < 2 or glon.size < 2:
            raise ValueError("need at least a 2x2 grid to interpolate")

        y = np.asarray(point_lat, dtype="f8")
        x = to_signed_lon(point_lon)

        def locate(g, p):
            i0 = np.clip(np.searchsorted(g, p, side="right") - 1, 0, g.size - 2)
            w = np.clip((p - g[i0]) / (g[i0 + 1] - g[i0]), 0.0, 1.0)
            return i0, w

        i0, wy = locate(glat, y)
        j0, wx = locate(glon, x)
        inside = (y >= glat[0]) & (y <= glat[-1]) & (x >= glon[0]) & (x <= glon[-1])
        return cls(lat_order, lon_order, i0, j0, wy, wx, inside, method, glat.size, glon.size)

    def apply(self, da: xr.DataArray, lat_dim: str = "lat", lon_dim: str = "lon") -> xr.DataArray:
        """Interpolate `da` to the points; (lat, lon) become a `station` axis
        placed last. Leading dims and their coordinates are preserved."""
        if da.sizes[lat_dim] != self.n_lat or da.sizes[lon_dim] != self.n_lon:
            raise ValueError(
                f"array grid ({da.sizes[lat_dim]}x{da.sizes[lon_dim]}) is not the grid this "
                f"interpolator was built for ({self.n_lat}x{self.n_lon})")
        lead = [d for d in da.dims if d not in (lat_dim, lon_dim)]
        v = da.transpose(*lead, lat_dim, lon_dim).values.astype("f8")
        v = v[..., self.lat_order, :][..., :, self.lon_order]

        i0, j0 = self.i0, self.j0
        if self.method == "nearest":
            out = v[..., i0 + (self.wy >= 0.5), j0 + (self.wx >= 0.5)]
        else:
            wy, wx = self.wy, self.wx
            out = ((1 - wy) * (1 - wx) * v[..., i0, j0] + (1 - wy) * wx * v[..., i0, j0 + 1]
                   + wy * (1 - wx) * v[..., i0 + 1, j0] + wy * wx * v[..., i0 + 1, j0 + 1])
        out = np.where(self.inside, out, np.nan).astype("f4")
        coords = {d: da.coords[d] for d in lead if d in da.coords}
        return xr.DataArray(out, dims=[*lead, "station"], coords=coords, name=da.name)


def interpolator_for(grid: xr.DataArray | xr.Dataset, stations: xr.Dataset, method="bilinear"):
    return GridInterpolator.build(grid["lat"].values, grid["lon"].values,
                                  stations["latitude"].values, stations["longitude"].values, method)


# ---------------------------------------------------------------------------
# curvilinear grids (CERRA's Lambert grid: lat/lon are 2-D)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurvilinearInterpolator:
    """Grid with 2-D lat/lon -> points, by the k nearest grid cells (k=1:
    nearest cell; k=4: inverse-distance weighting, close to bilinear at 5 km).
    Built once with a KD-tree on 3-D unit vectors, so distances are true
    great-circle distances and the prime meridian needs no special care."""

    flat_idx: np.ndarray      # (n_points, k) indices into the flattened (y, x) grid
    weights: np.ndarray       # (n_points, k)
    inside: np.ndarray        # False -> nearest cell further than max_km -> NaN
    shape: tuple[int, int]
    k: int

    @staticmethod
    def _xyz(lat, lon):
        la, lo = np.deg2rad(np.asarray(lat, "f8")), np.deg2rad(np.asarray(lon, "f8"))
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    @classmethod
    def build(cls, grid_lat2d, grid_lon2d, point_lat, point_lon, k: int = 4, max_km: float = 15.0):
        from scipy.spatial import cKDTree
        shape = np.asarray(grid_lat2d).shape
        tree = cKDTree(cls._xyz(np.ravel(grid_lat2d), np.ravel(grid_lon2d)))
        d, idx = tree.query(cls._xyz(point_lat, point_lon), k=k)
        d, idx = np.atleast_2d(d).reshape(len(np.atleast_1d(point_lat)), k), np.atleast_2d(idx).reshape(-1, k)
        km = 6371.0 * 2 * np.arcsin(np.clip(d / 2, 0, 1))          # chord -> great-circle
        inside = km[:, 0] <= max_km
        if k == 1:
            w = np.ones_like(km)
        else:
            w = 1.0 / np.maximum(km, 0.05) ** 2
            w /= w.sum(1, keepdims=True)
        return cls(idx, w, inside, (int(shape[0]), int(shape[1])), k)

    def apply(self, da: xr.DataArray, y_dim: str = "y", x_dim: str = "x") -> xr.DataArray:
        """(..., y, x) -> (..., station); leading dims and coords kept."""
        if (da.sizes[y_dim], da.sizes[x_dim]) != self.shape:
            raise ValueError(f"grid {(da.sizes[y_dim], da.sizes[x_dim])} is not the grid this interpolator was built for {self.shape}")
        lead = [d for d in da.dims if d not in (y_dim, x_dim)]
        v = da.transpose(*lead, y_dim, x_dim).values.reshape(*[da.sizes[d] for d in lead], -1)
        out = (v[..., self.flat_idx] * self.weights).sum(-1)
        out = np.where(self.inside, out, np.nan).astype("f4")
        coords = {d: da.coords[d] for d in lead if d in da.coords}
        return xr.DataArray(out, dims=[*lead, "station"], coords=coords, name=da.name)


def curvilinear_for(lat2d, lon2d, stations: xr.Dataset, k: int = 4) -> CurvilinearInterpolator:
    return CurvilinearInterpolator.build(lat2d, lon2d, stations["latitude"].values, stations["longitude"].values, k=k)


# ---------------------------------------------------------------------------
# lapse-rate correction
# ---------------------------------------------------------------------------
def lapse_correct(t, h_station, h_model, gamma: float = STD_LAPSE):
    """Move a model temperature from the model's surface height to the
    station's: T_station = T_model - gamma * (h_station - h_model).

    A station ABOVE the model surface is colder. gamma is a constant (the
    standard 6.5 K/km by default), which is exactly what fails in a
    heatwave night: valley inversions mean the real lapse rate is weaker,
    or reversed, so the correction over-cools high stations at night. That
    failure is the thing to measure by hour of day and elevation band, not
    a reason to tune gamma until it hides."""
    return t - gamma * (h_station - h_model)


def height_from_geopotential(z: xr.DataArray) -> xr.DataArray:
    """Surface geopotential (m^2 s^-2, ERA5 `z_sfc`) -> height in m."""
    return (z / G).rename("h_model")


def orography_on_grid(orography: xr.DataArray, like: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Cut a (possibly global) orography field to the grid of `like`.

    Nearest-with-tolerance rather than exact equality, so float noise in the
    coordinates of two separately produced archives cannot make this fail or,
    worse, silently pick a neighbour."""
    return orography.sel(lat=like["lat"].values, lon=like["lon"].values,
                         method="nearest", tolerance=1e-3)


# ---------------------------------------------------------------------------
# forecast -> stations
# ---------------------------------------------------------------------------
def downscale_to_stations(
    forecast: xr.Dataset,
    stations: xr.Dataset,
    variable: str = "t2m",
    orography: xr.DataArray | None = None,
    gamma: float = STD_LAPSE,
) -> xr.Dataset:
    """The baseline ladder, applied to a forecast zarr opened with xarray.

    `forecast` is as the backend writes it: (time, lead_time[, ensemble], lat, lon).
    Returns dims (init_time, lead_time[, ensemble], station) with variables
    `nearest`, `bilinear` and -- if `orography` (HEIGHT in m on the forecast
    grid) is given -- `bilinear_lapse`. `valid_time` is a (init_time, lead_time)
    coordinate."""
    fc = forecast[variable].rename({"time": "init_time"})
    bil = interpolator_for(fc, stations, "bilinear")
    near = interpolator_for(fc, stations, "nearest")
    if not bil.inside.all():
        logger.warning("%d station(s) fall outside the forecast grid -> NaN: %s",
                       int((~bil.inside).sum()), list(stations.station.values[~bil.inside][:5]))

    out = xr.Dataset({"nearest": near.apply(fc), "bilinear": bil.apply(fc)})
    if orography is not None:
        h_model = bil.apply(orography.astype("f8"))                    # (station,)
        h_station = stations["elevation"]
        out["bilinear_lapse"] = lapse_correct(out["bilinear"], h_station.values, h_model.values, gamma)
        out["h_model"] = ("station", h_model.values)
        out["h_diff"] = ("station", (h_station.values - h_model.values).astype("f4"))
    out = out.assign_coords(station=stations["station"].values)
    for c in ("latitude", "longitude", "elevation"):
        out = out.assign_coords({c: ("station", stations[c].values)})
    out = out.assign_coords(
        valid_time=(("init_time", "lead_time"),
                    out["init_time"].values[:, None] + out["lead_time"].values[None, :]))
    out.attrs.update({"variable": variable, "gamma_K_per_m": gamma,
                      "interpolation": "bilinear / nearest, no wrap"})
    return out


def analysis_baseline(
    source,
    forecast_like: xr.Dataset,
    stations: xr.Dataset,
    variable: str = "t2m",
    orography: xr.DataArray | None = None,
    gamma: float = STD_LAPSE,
) -> xr.Dataset:
    """ERA5 (or any analysis source with the Earth2Studio call signature)
    interpolated to the stations at each forecast VALID time.

    Dims (init_time, lead_time, station), same variable names as
    `downscale_to_stations` minus the ensemble, so the two line up for
    scoring. No forecast error is in here: the gap between these and the
    station observations is the representativeness floor."""
    init = forecast_like["time"].values
    lead = forecast_like["lead_time"].values
    valid = init[:, None] + lead[None, :]
    uniq = pd.DatetimeIndex(np.unique(valid.ravel()))
    logger.info("Fetching %d unique analysis time(s) for the baseline", uniq.size)

    da = source(list(uniq), [variable]).sel(variable=variable, drop=True)   # (time, lat, lon)
    da = da.sel(lat=forecast_like["lat"].values, lon=forecast_like["lon"].values,
                method="nearest", tolerance=1e-3)
    bil = interpolator_for(da, stations, "bilinear")
    near = interpolator_for(da, stations, "nearest")
    cols = {"nearest": near.apply(da), "bilinear": bil.apply(da)}
    if orography is not None:
        h_model = bil.apply(orography.astype("f8")).values
        cols["bilinear_lapse"] = lapse_correct(cols["bilinear"], stations["elevation"].values,
                                               h_model, gamma)

    pos = np.searchsorted(uniq.to_numpy().astype("datetime64[ns]"), valid.ravel().astype("datetime64[ns]"))
    out = {}
    for name, arr in cols.items():
        vals = arr.values[pos].reshape(init.size, lead.size, -1)
        out[name] = (("init_time", "lead_time", "station"), vals)
    return xr.Dataset(out, coords={
        "init_time": init, "lead_time": lead, "station": stations["station"].values,
        "valid_time": (("init_time", "lead_time"), valid)})


# ---------------------------------------------------------------------------
# pairing with observations
# ---------------------------------------------------------------------------
def observations_at_valid_times(fc_st: xr.Dataset, stations: xr.Dataset, variable: str = "t2m") -> xr.DataArray:
    """Observed values laid out like the forecast: (init_time, lead_time, station).

    A valid time with no observation row (a forecast step past the station
    window, or a gap) is NaN, so downstream scoring skips it instead of
    erroring. Stations are matched by id, not position."""
    obs = stations[variable].sel(station=fc_st["station"].values)
    valid = fc_st["valid_time"].values
    flat = obs.reindex(time=valid.ravel().astype("datetime64[ns]")).values      # (n_valid, station)
    vals = flat.reshape(*valid.shape, obs.sizes["station"])
    return xr.DataArray(vals, dims=("init_time", "lead_time", "station"),
                        coords={"init_time": fc_st["init_time"], "lead_time": fc_st["lead_time"],
                                "station": fc_st["station"], "valid_time": fc_st["valid_time"]},
                        name=f"obs_{variable}")