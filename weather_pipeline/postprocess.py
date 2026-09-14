"""
Convert a model's raw Earth2Studio Zarr output into the common analysis
schema every downstream step (plotting, scoring) expects: dims
(valid_time, latitude, longitude), one data variable per channel id
(e.g. ds["t2m"]), regardless of which model produced it.

Earth2Studio's ZarrBackend already writes one array per requested variable
name with dims (time, lead_time, lat, lon) -- so this is mostly a rename +
collapsing the (time, lead_time) pair into a single valid_time axis.
"""

from __future__ import annotations

import logging

import xarray as xr

logger = logging.getLogger(__name__)


def load_forecast(zarr_path: str, variables: list[str] | None = None) -> xr.Dataset:
    logger.info("Loading forecast from %s", zarr_path)
    ds = xr.open_zarr(zarr_path)

    # drop singleton batch/ensemble-style dims some models add
    for dim in ("batch", "ensemble"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)

    if variables is not None:
        ds = ds[variables]

    assert ds.sizes["time"] == 1, "load_forecast assumes a single init time per run"
    init_time = ds["time"].values[0]
    lead_time = ds["lead_time"].values  # timedelta64 array
    valid_time = init_time + lead_time

    ds = ds.isel(time=0, drop=True)
    ds = ds.rename({"lead_time": "valid_time"})
    # keep lead_time around too (handy for RMSE-by-lead-time plots)
    ds = ds.assign_coords(
        valid_time=("valid_time", valid_time),
        lead_time=("valid_time", lead_time),
    )
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    logger.info(
        "Loaded forecast: init_time=%s, %d lead times, variables=%s",
        init_time, ds.sizes["valid_time"], list(ds.data_vars),
    )
    return ds