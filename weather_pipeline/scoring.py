"""
On-demand RMSE scoring. Call `compute_rmse(forecast, truth, variable)` for
whichever variable you care about -- it's not computed for every variable
automatically, since that's wasted work most of the time.

`fetch_truth` reuses the same ArraylakeERA5Source used for initial
conditions, just pointed at the forecast's valid times instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from .datasource import ArraylakeERA5Source

logger = logging.getLogger(__name__)


def fetch_truth(
    source: ArraylakeERA5Source, variable: str, valid_times
) -> xr.Dataset:
    """Pulls ERA5 analysis for one variable at the given valid times, in the
    same (valid_time, latitude, longitude) schema as `postprocess.load_forecast`."""
    logger.info("Fetching ERA5 truth for '%s' at %d valid times", variable, len(valid_times))
    da = source(valid_times, [variable])  # dims: time, variable, lat, lon
    da = da.sel(variable=variable, drop=True)
    da = da.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})
    return da.to_dataset(name=variable)


def _latitude_weights(latitude: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(latitude))
    return weights / weights.mean()


def compute_rmse(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str
) -> pd.Series:
    """Latitude-weighted RMSE for one variable, indexed by lead time (hours)."""
    logger.info("Computing RMSE for '%s'", variable)
    fc = forecast_ds[variable]
    truth = truth_ds[variable].sel(valid_time=fc["valid_time"])
    weights = _latitude_weights(fc["latitude"])

    err2 = (fc - truth) ** 2
    rmse = np.sqrt(err2.weighted(weights).mean(dim=("latitude", "longitude")))

    lead_hours = (fc["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    result = pd.Series(
        rmse.values, index=pd.Index(lead_hours, name="lead_time_h"), name=f"rmse_{variable}"
    )
    logger.info("RMSE for '%s' computed at %d lead times", variable, len(result))
    return result


def compute_difference(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str
) -> xr.DataArray:
    """Forecast-minus-truth error field, one value per grid point per lead
    time -- i.e. the spatial map RMSE is a summary of, rather than the
    reduced RMSE number itself.

    Keeps the full (valid_time, latitude, longitude) resolution, so you can
    feed it straight into `plotting.render_rotating_gif(..., center=0)` to
    animate where and how the error grows with lead time."""
    logger.info("Computing forecast-minus-truth difference for '%s'", variable)
    fc = forecast_ds[variable]
    truth = truth_ds[variable].sel(valid_time=fc["valid_time"])
    diff = fc - truth
    diff.name = f"{variable}_error"
    return diff