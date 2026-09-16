"""
Earth2Studio-protocol DataSource backed by Brightband's ECMWF IFS analysis
on Arraylake.

Why this exists alongside ArraylakeERA5Source: the public ERA5 repo lags
real time by roughly a quarter, so recent init times simply are not in it.
The IFS repo is the other way round -- a rolling ~two-week window of recent
analyses and nothing historical. Between them they cover "any date" and
"the last fortnight" respectively, and which one you want depends entirely
on the init time.

It is also the analysis AIFS and Aurora 1.5 were fine-tuned on, so running
those models from IFS removes the initialisation mismatch that ERA5
initialisation imposes on them.

ROLLING WINDOW: forecasts made from this source cannot be reproduced later
once the window has moved past them. For a reproducible benchmark, either
archive the initial conditions yourself as they go by, or stay on ERA5 and
accept the mismatch.

Layout differences from the ERA5 repo, all handled here:
  * one dataset, not single/pressure groups
  * indexed by (init_time, lead_time) -- an analysis is lead_time=0
  * longitudes stored -180..180, so they need rolling to 0..360
  * static fields live on their own branch, not merged to main
  * GRIB-style short names (u10 rather than u10m, z_sfc rather than z)
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import xarray as xr

try:
    from earth2studio.data.utils import prep_data_inputs
except ModuleNotFoundError:  # see the note in datasource.py
    def prep_data_inputs(time, variable):
        if isinstance(time, (str, pd.Timestamp, np.datetime64)) or not hasattr(time, "__iter__"):
            time = [time]
        time = [pd.Timestamp(t) for t in time]
        variable = [variable] if isinstance(variable, str) else list(variable)
        return time, variable

logger = logging.getLogger(__name__)

IFS_REPO = "emmahf/ecmwf-ifs-initial-conditions-open-subscription"
STATIC_BRANCH = "add-static-vars"

# Earth2Studio channel id -> IFS repo variable name.
SURFACE_MAP: dict[str, list[str]] = {
    "u10m": ["u10"], "v10m": ["v10"], "d2m": ["d2m"], "t2m": ["t2m"],
    "msl": ["msl"], "skt": ["skt"], "sp": ["sp"],
    # the repo carries total column water; AIFS asks for tcw, and our ERA5
    # alias table spells the same slot "tcwv"
    "tcwv": ["tcw", "tcwv"],
    "stl1": ["stl1"], "stl2": ["stl2"],
    "swvl1": ["swvl1"], "swvl2": ["swvl2"],
    "sic": ["ci", "siconc", "sic"],
    "sd": ["sd"],
    "tcc": ["tcc"], "lcc": ["lcc"], "mcc": ["mcc"], "hcc": ["hcc"],
    "u100m": ["u100"], "v100m": ["v100"],
}
PRESSURE_PARAMS = {"z": "z", "q": "q", "t": "t", "u": "u", "v": "v", "w": "w"}

# anemoi static names -> repo names. z_sfc is SURFACE geopotential, which is
# a different field from pressure-level z.
STATIC_MAP: dict[str, list[str]] = {
    "z": ["z_sfc"], "lsm": ["lsm"], "sdor": ["sdor"], "slor": ["slor"],
}
LEVEL_DIM_CANDIDATES = ["level", "pressure_level", "isobaricInhPa"]


def _shift_lon(ds):
    if float(np.asarray(ds.longitude.values).min()) >= 0:
        return ds
    return ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")


@functools.lru_cache(maxsize=1)
def open_ifs_repo(repo_name: str = IFS_REPO, static_branch: str = STATIC_BRANCH):
    """Open the analysis and its statics. Cached: several round trips."""
    from arraylake import Client

    def _open(branch):
        repo = Client().get_repo(repo_name)
        return xr.open_zarr(repo.readonly_session(branch).store,
                            zarr_format=3, consolidated=False, chunks=None)

    logger.info("Opening IFS repo '%s'", repo_name)
    ds = _open("main")

    wanted = [c for cands in STATIC_MAP.values() for c in cands]
    if all(any(c in ds for c in cands) for cands in STATIC_MAP.values()):
        statics = ds
    else:
        logger.info("Statics not on main; reading branch '%s'", static_branch)
        statics = _open(static_branch)

    return _shift_lon(ds), _shift_lon(statics)


class ArraylakeIFSSource:
    """DataSource protocol: __call__(time, variable) -> (time, variable, lat, lon).

    Serves both as a model's initial-condition source and as verification
    truth, same as ArraylakeERA5Source -- the only difference is which
    timestamps it can answer for."""

    def __init__(self, ds: xr.Dataset | None = None, statics: xr.Dataset | None = None):
        if ds is None:
            ds, statics = open_ifs_repo()
        self.ds = ds
        self._statics = statics
        self.level_dim = next(
            (d for d in LEVEL_DIM_CANDIDATES if d in ds.dims), "level"
        )
        times = np.sort(ds.init_time.values)
        # the first stored time has no predecessor, so it cannot serve a
        # model that needs the step before the init time (AIFS does)
        self.available_times = times[1:]
        logger.info(
            "IFS window: %s .. %s (%d usable init times)",
            self.available_times[0], self.available_times[-1], len(self.available_times),
        )

    def check_covers(self, times) -> None:
        missing = [t for t in pd.to_datetime(list(times))
                   if np.datetime64(t) not in self.available_times]
        if missing:
            raise KeyError(
                f"IFS analysis does not cover {[str(t) for t in missing]}. "
                f"Its rolling window is {self.available_times[0]} .. "
                f"{self.available_times[-1]}. Use a recent init time, or "
                "switch to the ERA5 source for historical dates."
            )

    def _resolve(self, v: str):
        """-> ('surface', name) | ('pressure', name, level)."""
        if v in SURFACE_MAP:
            name = next((c for c in SURFACE_MAP[v] if c in self.ds), None)
            if name:
                return ("surface", name)
            raise KeyError(
                f"'{v}' maps to {SURFACE_MAP[v]}, none present in the IFS repo. "
                f"Available: {sorted(self.ds.data_vars)}"
            )
        for prefix, repo_name in PRESSURE_PARAMS.items():
            if v.startswith(prefix) and v[len(prefix):].isdigit():
                if repo_name in self.ds:
                    return ("pressure", repo_name, int(v[len(prefix):]))
                raise KeyError(f"pressure variable '{repo_name}' not in the IFS repo")
        raise KeyError(
            f"Don't know how to resolve '{v}' against the IFS repo -- add it "
            "to SURFACE_MAP / PRESSURE_PARAMS in ifs_datasource.py. "
            f"Available: {sorted(self.ds.data_vars)}"
        )

    def get_statics(self) -> dict[str, xr.DataArray]:
        """The four constant fields AIFS needs, named as anemoi expects."""
        out = {}
        for name, cands in STATIC_MAP.items():
            match = next((c for c in cands if c in self._statics), None)
            if match is None:
                raise KeyError(
                    f"Static field '{name}' (tried {cands}) not found in the "
                    f"IFS repo. Available: {sorted(self._statics.data_vars)}"
                )
            da = self._statics[match]
            for dim in ("init_time", "lead_time", "time", "step"):
                if dim in da.dims:
                    da = da.isel({dim: 0}, drop=True)
            out[name] = da.rename({"latitude": "lat", "longitude": "lon"}).load()
        return out

    def __call__(self, time, variable) -> xr.DataArray:
        time, variable = prep_data_inputs(time, variable)
        self.check_covers(time)
        logger.info(
            "Fetching %d IFS variable(s) for %d timestamp(s): %s",
            len(variable), len(time), list(variable),
        )
        resolved = [self._resolve(v) for v in variable]

        arrays = []
        for v, spec in zip(variable, resolved):
            # an analysis is lead_time 0 of its own init_time
            sel = dict(init_time=[np.datetime64(t) for t in time],
                       lead_time=np.timedelta64(0, "ns"))
            if spec[0] == "surface":
                da = self.ds[spec[1]].sel(**sel, drop=True)
            else:
                da = self.ds[spec[1]].sel(**sel, **{self.level_dim: spec[2]}, drop=True)
            da = da.rename({"init_time": "time",
                            "latitude": "lat", "longitude": "lon"}).load()
            extra = [c for c in da.coords if c not in ("time", "lat", "lon")]
            da = da.drop_vars(extra)
            arrays.append(da.expand_dims(variable=[v]))

        out = xr.concat(arrays, dim="variable")
        logger.info("Fetched %s", dict(out.sizes))
        return out.transpose("time", "variable", "lat", "lon")