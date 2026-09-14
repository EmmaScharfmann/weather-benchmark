"""
Generic Earth2Studio DataSource backed by an Arraylake ERA5 repo.

This is intentionally model-agnostic: it doesn't hardcode a fixed variable
list. It resolves whatever Earth2Studio channel ids a model asks for (e.g.
"t2m", "z500", "q850") against the Arraylake single-level / pressure-level
datasets, using an alias table you can extend. That's what lets the same
DataSource serve Aurora, AIFS, Atlas, or anything else without changes --
each model's `input_coords()["variable"]` just drives what gets requested.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from arraylake import Client

from earth2studio.data.utils import prep_data_inputs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias tables: Earth2Studio channel id -> candidate variable names in the
# Arraylake ERA5 repo. If a model requests an id that isn't resolvable, the
# error message names exactly which id failed -- add an alias here to fix it.
# ---------------------------------------------------------------------------
SURFACE_ALIASES: dict[str, list[str]] = {
    "u10m": ["u10", "10m_u_component_of_wind", "10u"],
    "v10m": ["v10", "10m_v_component_of_wind", "10v"],
    "u100m": ["u100", "100m_u_component_of_wind"],
    "v100m": ["v100", "100m_v_component_of_wind"],
    "t2m": ["t2m", "2m_temperature", "2t"],
    "d2m": ["d2m", "2m_dewpoint_temperature"],
    "msl": ["msl", "mean_sea_level_pressure"],
    "sp": ["sp", "surface_pressure"],
    "tcwv": ["tcwv", "total_column_water_vapour", "total_column_water_vapor"],
    "tp": ["tp", "total_precipitation"],
    "sst": ["sst", "sea_surface_temperature"],
    "lsm": ["lsm", "land_sea_mask"],
}
PRESSURE_ALIASES: dict[str, list[str]] = {
    "z": ["z", "geopotential"],
    "q": ["q", "specific_humidity"],
    "t": ["t", "temperature"],
    "u": ["u", "u_component_of_wind"],
    "v": ["v", "v_component_of_wind"],
    "w": ["w", "vertical_velocity"],
}
DIM_RENAME = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
LEVEL_DIM_CANDIDATES = ["level", "pressure_level", "isobaricInhPa"]

# Variables that don't exist directly in the repo and have to be derived
# from other variables. Each entry names a method on ArraylakeERA5Source
# taking a single target timestamp and returning a DataArray with a "time"
# dim of length 1 for that timestamp.
DERIVED_SURFACE: dict[str, str] = {
    "tp06": "_fetch_tp06",
}


def _first_match(names: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"None of {candidates} found in dataset variables {names}")


def open_era5_repo(
    repo_name: str = "earthmover-public/era5",
    branch: str = "main",
    single_group: str = "single/spatial",
    pressure_group: str = "pressure/spatial",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Open the Arraylake ERA5 repo lazily. Uses chunks=None (plain
    zarr-backed indexing, no dask) so nothing is read until a specific
    timestamp/variable is actually requested -- same pattern as your
    original snippet, and much cheaper than dask for "grab a few
    timestamps out of a huge archive" access patterns."""
    client = Client()
    repo = client.get_repo(repo_name)  # requires pcodec
    session = repo.readonly_session(branch)
    logger.info("Opening Arraylake repo '%s' (branch=%s)...", repo_name, branch)
    single = xr.open_zarr(session.store, group=single_group, chunks=None)
    pl = xr.open_zarr(session.store, group=pressure_group, chunks=None)
    logger.info(
        "Opened '%s' groups '%s' and '%s' (lazy -- nothing read yet)",
        repo_name, single_group, pressure_group,
    )
    return single, pl


class ArraylakeERA5Source:
    """Earth2Studio DataSource wrapping Arraylake ERA5 single+pressure datasets.

    Implements the DataSource protocol:
        __call__(time, variable) -> xr.DataArray, dims [time, variable, lat, lon]

    Works both as the model's initial-condition source AND as a verification
    ("truth") source for RMSE, since both just call it with different times.

    Important: lat/lon standardization (descending lat, 0-360 lon) is only
    ever applied AFTER time/level selection has already shrunk the array to
    a couple of timesteps -- never on the full multi-year dataset. Doing it
    upfront on the whole archive is what makes construction "hang".
    """

    def __init__(self, single_ds: xr.Dataset, pl_ds: xr.Dataset):
        self.single = single_ds.rename(
            {k: v for k, v in DIM_RENAME.items() if k in single_ds.dims}
        )
        self.pl = pl_ds.rename({k: v for k, v in DIM_RENAME.items() if k in pl_ds.dims})
        self.level_dim = _first_match(list(self.pl.dims), LEVEL_DIM_CANDIDATES)

        # Cheap: just reads the 1D lat/lon coordinate arrays themselves,
        # not any of the actual data variables.
        self._flip_lat = bool(self.single["lat"].values[0] < self.single["lat"].values[-1])
        self._shift_lon = bool(self.single["lon"].values.min() < 0)
        logger.info(
            "Grid orientation detected: flip_lat=%s, shift_lon_to_0_360=%s",
            self._flip_lat, self._shift_lon,
        )

    def _fix_grid(self, da: xr.DataArray) -> xr.DataArray:
        """Applied to an already-small (few-timestep) slice only."""
        if self._flip_lat:
            da = da.isel(lat=slice(None, None, -1))
        if self._shift_lon:
            da = da.assign_coords(lon=(da["lon"] % 360)).sortby("lon")
        return da

    def _fetch_tp06(self, target_time) -> xr.DataArray:
        """6h-accumulated precip ending at target_time, built from 6 hourly
        ERA5 'tp' accumulations. Pulls straight from the full (un-sliced)
        single-level dataset since it needs the 5 hours *before* whichever
        timestamp was requested, not just that timestamp itself."""
        tp_name = _first_match(list(self.single.data_vars), SURFACE_ALIASES["tp"])
        end = pd.Timestamp(target_time)
        hourly_times = pd.date_range(end - pd.Timedelta(hours=5), end, freq="1h")
        logger.debug("  tp06 @ %s <- summing hourly tp %s..%s", end, hourly_times[0], hourly_times[-1])
        tp_hourly = self.single[tp_name].sel(time=hourly_times)
        tp06 = tp_hourly.sum(dim="time")
        tp06 = tp06.expand_dims(time=[np.datetime64(end)])
        return tp06

    def __call__(self, time, variable) -> xr.DataArray:
        time, variable = prep_data_inputs(time, variable)
        logger.info(
            "Fetching %d variable(s) for %d timestamp(s): %s",
            len(variable), len(time), list(variable),
        )
        arrays = []
        for v in variable:
            if v in DERIVED_SURFACE:
                method = getattr(self, DERIVED_SURFACE[v])
                da = xr.concat([method(t) for t in time], dim="time")
            elif v in SURFACE_ALIASES:
                name = _first_match(list(self.single.data_vars), SURFACE_ALIASES[v])
                logger.debug("  %s -> single/%s", v, name)
                da = self.single[name].sel(time=time)
            else:
                matched = False
                for prefix, cands in PRESSURE_ALIASES.items():
                    if v.startswith(prefix) and v[len(prefix):].isdigit():
                        level = int(v[len(prefix):])
                        name = _first_match(list(self.pl.data_vars), cands)
                        logger.debug("  %s -> pressure/%s @ %s hPa", v, name, level)
                        da = self.pl[name].sel(time=time, **{self.level_dim: level})
                        matched = True
                        break
                if not matched:
                    raise KeyError(
                        f"Don't know how to resolve Earth2Studio variable id "
                        f"'{v}' against the Arraylake ERA5 repo -- add an "
                        "alias to SURFACE_ALIASES/PRESSURE_ALIASES, or a "
                        "method to DERIVED_SURFACE if it needs to be computed."
                    )
            da = da.load()  # pull only the requested slice into memory
            da = self._fix_grid(da)
            # Drop leftover scalar coords (e.g. the selected pressure level)
            # that only exist on some variables -- xr.concat requires the
            # coordinate set to match across everything being concatenated.
            extra_coords = [c for c in da.coords if c not in ("time", "lat", "lon")]
            da = da.drop_vars(extra_coords)
            arrays.append(da.expand_dims(variable=[v]))
        out = xr.concat(arrays, dim="variable")
        logger.info("Fetched %s", dict(out.sizes))
        return out.transpose("time", "variable", "lat", "lon")