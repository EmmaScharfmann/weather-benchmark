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

try:
    from earth2studio.data.utils import prep_data_inputs
except ModuleNotFoundError:
    # ArraylakeERA5Source is used two ways: (a) as a model's initial-condition
    # source, which only ever runs inside that model's own isolated uv env
    # (earth2studio installed there -- the import above succeeds, and we use
    # earth2studio's own normalization exactly, matching what the model
    # expects); and (b) as a truth source for scoring, which runs in the
    # shared base env (earth2studio deliberately NOT installed there, so
    # isolated model envs stay isolated). This fallback covers case (b) only:
    # a light reimplementation of the same time/variable list-ification, used
    # solely for our own truth-fetching calls -- never for anything actually
    # fed into a model's forward pass. If you change how earth2studio's
    # prep_data_inputs normalizes inputs, mirror the change here too.
    def prep_data_inputs(time, variable):
        if isinstance(time, (str, pd.Timestamp, np.datetime64)) or not hasattr(time, "__iter__"):
            time = [time]
        time = [pd.Timestamp(t) for t in time]
        variable = [variable] if isinstance(variable, str) else list(variable)
        return time, variable

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
    # --- added for AIFS/anemoi, which asks for more surface fields than
    # the Earth2Studio models did. Skin temperature is NOT sea-surface
    # temperature -- keep them distinct.
    "skt": ["skt", "skin_temperature"],
    "cp": ["cp", "convective_precipitation"],
    "sd": ["sd", "snow_depth"],
    "stl1": ["stl1", "soil_temperature_level_1"],
    "stl2": ["stl2", "soil_temperature_level_2"],
    "stl3": ["stl3", "soil_temperature_level_3"],
    "stl4": ["stl4", "soil_temperature_level_4"],
    "swvl1": ["swvl1", "volumetric_soil_water_layer_1"],
    "swvl2": ["swvl2", "volumetric_soil_water_layer_2"],
    "swvl3": ["swvl3", "volumetric_soil_water_layer_3"],
    "swvl4": ["swvl4", "volumetric_soil_water_layer_4"],
    # Static surface fields AIFS needs in its input state. "z_surface" is
    # SURFACE geopotential from the single-level group -- deliberately a
    # different id from pressure-level "z500" etc, which PRESSURE_ALIASES
    # handles, so the two can't be confused.
    "z_surface": ["z", "geopotential"],
    "sdor": ["sdor", "standard_deviation_of_orography"],
    "slor": ["slor", "slope_of_sub_gridscale_orography"],
    # --- added for Aurora 1.5, whose input set is far wider than Aurora 1's
    # (26 surface variables vs 4). Names below match what the Arraylake ERA5
    # single-level group actually carries.
    "tcc": ["tcc", "total_cloud_cover"],
    "lcc": ["lcc", "low_cloud_cover"],
    "mcc": ["mcc", "medium_cloud_cover"],
    "hcc": ["hcc", "high_cloud_cover"],
    "blh": ["blh", "boundary_layer_height"],
    "cape": ["cape", "convective_available_potential_energy"],
    "fg10": ["fg10", "10m_wind_gust_since_previous_post_processing"],
    "zust": ["zust", "friction_velocity"],
    "fsr": ["fsr", "forecast_surface_roughness"],
    "lsp": ["lsp", "large_scale_precipitation"],
    "sf": ["sf", "snowfall"],
    "ssr": ["ssr", "surface_net_solar_radiation"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "tsr": ["tsr", "top_net_solar_radiation"],
    "tisr": ["tisr", "toa_incident_solar_radiation"],
    "fdir": ["fdir", "total_sky_direct_solar_radiation_at_surface"],
    "slhf": ["slhf", "surface_latent_heat_flux"],
    "ie": ["ie", "instantaneous_moisture_flux"],
    "tcw": ["tcw", "total_column_water"],
    # Sea-ice cover: spelled several ways across ERA5 products. If none
    # resolve, the repo genuinely lacks it -- see the note in the error.
    "sic": ["sic", "ci", "siconc", "sea_ice_cover", "sea_ice_area_fraction"],
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
    # Sea-ice cover is absent from this repo but required by Aurora 1.5 as
    # a time-varying input. Pulled from the public ARCO-ERA5 mirror and
    # cached locally -- see weather_pipeline/sea_ice.py.
    "sic": "_fetch_sic",

}



def _first_match(names: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"None of {candidates} found in dataset variables {names}")


def _names_incl_coords(ds: xr.Dataset) -> list[str]:
    """data_vars AND coords.

    The four constant fields AIFS needs (lsm, z_sfc, slor, sdor) are stored
    in the Arraylake ERA5 repo as 2-D COORDINATES on both groups, not as
    data variables -- so a data_vars-only lookup reports them as missing
    even though they're right there."""
    return list(ds.data_vars) + list(ds.coords)


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
        tp_name = _first_match(_names_incl_coords(self.single), SURFACE_ALIASES["tp"])
        end = pd.Timestamp(target_time)
        hourly_times = pd.date_range(end - pd.Timedelta(hours=5), end, freq="1h")
        logger.debug("  tp06 @ %s <- summing hourly tp %s..%s", end, hourly_times[0], hourly_times[-1])
        tp_hourly = self.single[tp_name].sel(time=hourly_times)
        tp06 = tp_hourly.sum(dim="time")
        tp06 = tp06.expand_dims(time=[np.datetime64(end)])
        return tp06

    def _fetch_sic(self, target_time) -> xr.DataArray:
        """Sea-ice cover for one timestamp, from ARCO-ERA5.

        Same contract as the other DERIVED_SURFACE methods: one timestamp
        in, a DataArray with a length-1 time dim out, so __call__ can
        concatenate it alongside fields read from this repo.

        Imported lazily so that gcsfs is only needed in environments that
        actually run Aurora 1.5.
        """
        from .sea_ice import fetch_sea_ice

        return fetch_sea_ice([target_time]).rename(None)

    def _check_resolvable(self, variable) -> None:
        """Raise once, naming every id that cannot be resolved, plus what
        the repo actually holds."""
        single_names = _names_incl_coords(self.single)
        pl_names = _names_incl_coords(self.pl)
        unresolved: dict[str, list[str]] = {}

        for v in variable:
            if v in DERIVED_SURFACE:
                continue
            if v in SURFACE_ALIASES:
                if not any(c in single_names for c in SURFACE_ALIASES[v]):
                    unresolved[v] = SURFACE_ALIASES[v]
                continue
            for prefix, cands in PRESSURE_ALIASES.items():
                if v.startswith(prefix) and v[len(prefix):].isdigit():
                    if not any(c in pl_names for c in cands):
                        unresolved[v] = cands
                    break
            else:
                unresolved[v] = ["<no alias defined>"]

        if not unresolved:
            return
        raise KeyError(
            f"{len(unresolved)} requested variable(s) cannot be resolved "
            f"against the Arraylake ERA5 repo:\n"
            + "\n".join(f"    {v}: tried {c}" for v, c in sorted(unresolved.items()))
            + "\n\nSingle-level group holds: "
            + f"{sorted(single_names)}\n\n"
            "Fix by adding the repo's own spelling to SURFACE_ALIASES / "
            "PRESSURE_ALIASES. If a field is genuinely absent from the repo "
            "(sea-ice cover, for instance, is not in every ERA5 subset), the "
            "model cannot be initialised from this source and needs a "
            "different one."
        )

    def __call__(self, time, variable) -> xr.DataArray:
        time, variable = prep_data_inputs(time, variable)
        logger.info(
            "Fetching %d variable(s) for %d timestamp(s): %s",
            len(variable), len(time), list(variable),
        )
        # Resolve every requested id to a concrete repo variable BEFORE
        # fetching anything, so one run reports all the unresolvable ones
        # together. Failing on the first means discovering a long input list
        # (Aurora 1.5 asks for 83) one variable per run.
        self._check_resolvable(variable)

        arrays = []
        for v in variable:
            if v in DERIVED_SURFACE:
                method = getattr(self, DERIVED_SURFACE[v])
                da = xr.concat([method(t) for t in time], dim="time")
            elif v in SURFACE_ALIASES:
                name = _first_match(_names_incl_coords(self.single), SURFACE_ALIASES[v])
                logger.debug("  %s -> single/%s", v, name)
                da = self.single[name].sel(time=time)
            else:
                matched = False
                for prefix, cands in PRESSURE_ALIASES.items():
                    if v.startswith(prefix) and v[len(prefix):].isdigit():
                        level = int(v[len(prefix):])
                        name = _first_match(_names_incl_coords(self.pl), cands)
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


def build_data_source(source: str = "era5"):
    """ERA5 for historical dates, IFS for recent ones.

    Lives here rather than in pipeline.py on purpose: pipeline imports
    backends, which imports torch, and the base environment (where scoring
    and reporting run) has no torch. Keeping the source factory in this
    module lets run_benchmark build a data source without pulling in the
    model stack.

    The two archives barely overlap -- the public ERA5 repo lags real time
    by about a quarter, while the IFS repo is a rolling ~two-week window --
    so which one you want is decided by the init time. IFS is also the
    analysis AIFS and Aurora 1.5 were fine-tuned on, at the cost of
    forecasts that cannot be reproduced once the window moves on.
    """
    if source == "ifs":
        from .ifs_datasource import ArraylakeIFSSource
        return ArraylakeIFSSource()
    single, pl = open_era5_repo()
    return ArraylakeERA5Source(single, pl)