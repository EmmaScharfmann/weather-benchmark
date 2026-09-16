"""
Sea-ice cover from ARCO-ERA5.

Aurora 1.5 requires `ci` (sea-ice cover) as a time-varying surface input.
It is the ONLY one of its 19 input variables missing from the Arraylake
ERA5 repo -- everything else, including snow depth, is there. AIFS and
Aurora 1 never ask for it, which is why they run without this module.

So this one field comes from ARCO-ERA5, the publicly readable Google-hosted
ERA5 mirror: anonymous access, no CDS account, same 0.25 degree grid, and
chunked one timestep per chunk so a single hour is a cheap read.

Fetched values are cached to a local zarr keyed by timestamp, so repeated
runs at the same init time hit disk instead of the network.

CAVEAT: this stitches two archives together. Everything else comes from the
Arraylake ERA5 repo; sea ice comes from ARCO-ERA5. Both are ERA5, so they
should agree, but they are separately produced copies. If Aurora 1.5's
polar behaviour ever looks odd, this seam is the first thing to check.
"""

from __future__ import annotations

import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CACHE_DIR = Path("outputs/cache/sea_ice")
VARIABLE = "sea_ice_cover"


def _open_arco() -> xr.Dataset:
    """Anonymous read of the public store.

    The store holds ~250 variables over decades at hourly resolution, so
    reading its metadata is the slow part -- not the data slice, which is a
    single chunk. drop_variables keeps the metadata parse to what we need;
    the local cache means this is rarely reached at all.
    """
    return xr.open_zarr(
        ARCO_ERA5,
        chunks=None,
        consolidated=True,
        storage_options={"token": "anon"},
    )


def _standardise(da: xr.DataArray) -> xr.DataArray:
    """ARCO-ERA5 names its dims latitude/longitude; the pipeline wants
    lat/lon, descending latitude, 0-360 longitude -- matching what
    ArraylakeERA5Source produces, so the field lines up with everything
    else in the input state."""
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
           if k in da.dims}
    if ren:
        da = da.rename(ren)
    if float(da["lat"].values[0]) < float(da["lat"].values[-1]):
        da = da.isel(lat=slice(None, None, -1))
    if float(np.asarray(da["lon"].values).min()) < 0:
        da = da.assign_coords(lon=(da["lon"] % 360)).sortby("lon")
    for dim in ("time", "step", "number"):
        if dim in da.dims and da.sizes.get(dim) == 1:
            da = da.isel({dim: 0}, drop=True)
    da.encoding.clear()
    for c in da.coords.values():
        c.encoding.clear()
    return da.fillna(0.0).astype("float32")
    


def fetch_sea_ice(times, cache_dir: Path | str | None = None) -> xr.DataArray:
    """Sea-ice cover at the given timestamps, dims (time, lat, lon).

    Cached per timestamp: the first call for a date pays the network, every
    later one reads a ~4 MB local zarr.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    times = [pd.Timestamp(t) for t in times]
    frames: dict[pd.Timestamp, xr.DataArray] = {}
    missing: list[pd.Timestamp] = []

    for t in times:
        path = cache_dir / f"{t.strftime('%Y%m%dT%H%M')}.zarr"
        if path.exists():
            try:
                frames[t] = xr.open_zarr(path, consolidated=False)[VARIABLE].load()
                continue
            except (KeyError, OSError, ValueError):
                # A store left behind by a crashed write exists on disk but
                # contains no arrays. Directory existence alone is not proof
                # of a complete cache entry, so discard and re-fetch.
                logger.warning("Discarding incomplete sea-ice cache at %s", path)
                shutil.rmtree(path, ignore_errors=True)
        missing.append(t)

    if missing:
        logger.info(
            "Fetching sea-ice cover from ARCO-ERA5 for %d timestamp(s): %s "
            "(first call also parses the store metadata, which is the slow part)",
            len(missing), [str(t) for t in missing],
        )
        ds = _open_arco()
        for t in missing:
            try:
                da = ds[VARIABLE].sel(time=np.datetime64(t)).load()
            except KeyError as exc:
                raise KeyError(
                    f"ARCO-ERA5 has no sea-ice cover at {t}. That archive lags "
                    "real time by a few months, so very recent dates are not in "
                    "it. Aurora 1.5 cannot be initialised for this date."
                ) from exc

            da = _standardise(da)
            path = cache_dir / f"{t.strftime('%Y%m%dT%H%M')}.zarr"

            # Drop the encoding inherited from ARCO-ERA5: it carries that
            # store's zarr-v2 Blosc compressor object, which zarr v3 refuses
            # with "Expected a BytesBytesCodec". We want a fresh local array,
            # not a copy of the source's chunk/codec layout.
            ds_out = da.to_dataset(name=VARIABLE)
            for v in ds_out.variables:
                ds_out[v].encoding.clear()

            # Write to a temp path and rename into place, so an interrupted
            # write can never leave behind a directory that looks like a
            # valid cache entry but holds nothing.
            tmp = path.with_name(path.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            ds_out.to_zarr(tmp, mode="w", zarr_format=3, consolidated=False)
            tmp.rename(path)

            frames[t] = da
            logger.info("  cached %s", path)

    out = xr.concat([frames[t] for t in times], dim="time")
    return out.assign_coords(time=[np.datetime64(t) for t in times])