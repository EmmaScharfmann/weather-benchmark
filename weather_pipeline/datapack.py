"""
The tutorial data pack: everything a reader needs, in one Hugging Face
dataset, with no GPU, no Arraylake account and no hour-long station download.

    export_pack(...)          builds the folder (run once, by us)
    load_pack(path_or_repo)   what notebooks call

Layout of the pack:

    README.md                          dataset card (sources, licenses, caveats)
    manifest.json                      provenance: model, inits, window, counts, versions
    forecasts/<model>_<var>.zarr       (time, lead_time[, ensemble], lat, lon) + orography(lat, lon)
    analysis/era5_<var>.zarr           (time, lat, lon) -- for the representativeness floor
    stations/metadata.parquet          one row per station
    stations/observations.parquet      long table: station_id, time, <var>...

Grid conventions INSIDE the pack (simpler than the pipeline's): longitudes
signed and ascending (-180..180), latitudes descending, `time` = init time on
forecasts (the backend's own schema). load_pack returns objects that drop
straight into station_interp / station_scoring:

    pack = load_pack("you/aifs-france-heatwave")
    fc_st = downscale_to_stations(pack.forecast, pack.stations, "t2m", orography=pack.orography)
    an    = analysis_baseline(pack.analysis, pack.forecast, pack.stations, "t2m", orography=pack.orography)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import metadata as _md
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .regions import to_signed_lon

logger = logging.getLogger(__name__)

PACK_VERSION = "0.1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _signed_sorted(da):
    """0..360 (possibly non-monotonic) -> signed, ascending longitudes."""
    return da.assign_coords(lon=to_signed_lon(da["lon"].values)).sortby("lon")


def _clean_encoding(ds):
    for v in ds.variables.values():
        v.encoding.clear()
    return ds


def _versions():
    out = {}
    for pkg in ("xarray", "zarr", "numpy", "pandas", "anemoi-inference", "wetterdienst", "weatherbenchX"):
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            pass
    return out


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export_pack(
    zarr_paths: Sequence[str | Path],
    stations: xr.Dataset,
    orography: xr.DataArray,
    analysis_source,
    out_dir: str | Path,
    variable: str = "t2m",
    model: str = "aifs-ens1.0",
    model_repo: str = "ecmwf/aifs-ens-1.0",
    caveats: Sequence[str] = (),
    overwrite: bool = False,
) -> Path:
    """Write the pack folder.

    zarr_paths: per-init forecast zarrs written by the backend (same grid).
    stations:   the Dataset from stations.load_stations (already QC'd).
    orography:  model surface HEIGHT (m), any grid covering the forecast's.
    analysis_source: Earth2Studio-style source (e.g. ArraylakeERA5Source);
                fetched at every forecast valid time and stored, so readers
                never need it.
    caveats:    extra lines for the dataset card (e.g. in-sample warnings)."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; pass overwrite=True")
    for sub in ("forecasts", "analysis", "stations"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # -- forecasts: concat inits, signed lon, add orography --------------------
    parts = [xr.open_zarr(p)[[variable]] for p in sorted(map(str, zarr_paths))]
    ref = parts[0]
    for p, ds in zip(zarr_paths, parts):
        if not (np.array_equal(ds.lat, ref.lat) and np.array_equal(ds.lon, ref.lon)
                and np.array_equal(ds.lead_time, ref.lead_time)):
            raise ValueError(f"{p}: grid or lead times differ from the first zarr")
    fc = xr.concat(parts, dim="time").sortby("time").load()
    fc = _signed_sorted(fc)
    oro = _signed_sorted(orography).sel(lat=fc.lat.values, lon=fc.lon.values, method="nearest", tolerance=1e-3)
    fc["orography"] = (("lat", "lon"), oro.values.astype("f4"))
    fc["orography"].attrs.update(units="m", long_name="model surface height (ERA5 z / g)")
    fc[variable].attrs.setdefault("units", "K")
    fc.attrs = {"model": model, "source_repo": model_repo, "variable": variable,
                "lon_convention": "signed, ascending", "time": "initialisation time (UTC)"}
    fname = f"{model}_{variable}.zarr"
    _clean_encoding(fc).to_zarr(out / "forecasts" / fname, mode="w")
    logger.info("forecasts: %s %s", fname, dict(fc[variable].sizes))

    # -- analysis at every valid time -------------------------------------------
    valid = (fc.time.values[:, None] + fc.lead_time.values[None, :]).ravel()
    times = pd.DatetimeIndex(np.unique(valid))
    # Fetched in batches and cropped immediately: the source returns GLOBAL
    # fields, and ~120 valid times at 0.25 deg would be ~0.5 GB at once.
    crops = []
    for i in range(0, times.size, 24):
        batch = analysis_source(list(times[i:i + 24]), [variable]).sel(variable=variable, drop=True)
        crops.append(_signed_sorted(batch).sel(lat=fc.lat.values, lon=fc.lon.values,
                                               method="nearest", tolerance=1e-3).load())
    an = xr.concat(crops, dim="time")
    an = an.assign_coords(lat=fc.lat.values, lon=fc.lon.values).astype("f4")
    an_ds = an.to_dataset(name=variable)
    an_ds[variable].attrs["units"] = "K"
    an_ds.attrs = {"source": "ERA5 reanalysis", "note": "analysis at every forecast valid time"}
    _clean_encoding(an_ds).to_zarr(out / "analysis" / f"era5_{variable}.zarr", mode="w")
    logger.info("analysis: %d time(s)", times.size)

    # -- stations: parquet, viewer-friendly ------------------------------------
    meta = pd.DataFrame({
        "station_id": stations.station.values.astype(str),
        "name": stations["name"].values.astype(str),
        "latitude": stations.latitude.values, "longitude": stations.longitude.values,
        "elevation_m": stations.elevation.values,
        "holdout": stations.holdout.values.astype(bool),
    })
    meta.to_parquet(out / "stations" / "metadata.parquet", index=False)
    obs_vars = [v for v in stations.data_vars if "time" in stations[v].dims and "station" in stations[v].dims]
    long = (stations[obs_vars].to_dataframe().reset_index()
            .rename(columns={"station": "station_id"})
            .dropna(subset=obs_vars, how="all"))
    long["station_id"] = long["station_id"].astype(str)
    long = long[["station_id", "time", *obs_vars]]
    long.to_parquet(out / "stations" / "observations.parquet", index=False)
    logger.info("stations: %d, observation rows: %d", len(meta), len(long))

    # -- manifest + card -------------------------------------------------------
    manifest = {
        "pack_version": PACK_VERSION, "created_utc": pd.Timestamp.utcnow().isoformat(),
        "model": model, "model_repo": model_repo, "variable": variable,
        "inits": [str(pd.Timestamp(t)) for t in fc.time.values],
        "lead_hours": [int(h) for h in fc.lead_time.values / np.timedelta64(1, "h")],
        "members": int(fc.sizes.get("ensemble", 1)),
        "grid": {"lat": [float(fc.lat.min()), float(fc.lat.max())], "lon": [float(fc.lon.min()), float(fc.lon.max())],
                 "shape": [int(fc.sizes["lat"]), int(fc.sizes["lon"])]},
        "observation_window": [str(pd.Timestamp(stations.time.values[0])), str(pd.Timestamp(stations.time.values[-1]))],
        "stations": int(len(meta)), "holdout_stations": int(meta.holdout.sum()),
        "station_provider": stations.attrs.get("provider", ""), "station_license": stations.attrs.get("license", ""),
        "caveats": list(caveats), "versions": _versions(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(_dataset_card(manifest, fname, variable))
    (out / "LICENSE").write_text(_LICENSE_TEXT.format(model_repo=model_repo,
                                                      station_license=manifest["station_license"]))
    logger.info("pack written to %s", out)
    return out


_LICENSE_TEXT = """This dataset combines material under different terms. Each part keeps its own license.

1. forecasts/: produced with {model_repo} (ECMWF). The model is released under
   CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Attribute ECMWF.

2. analysis/ and forecasts/*/orography: derived from ERA5, Copernicus Climate Change
   Service. "Contains modified Copernicus Climate Change Service information."
   Licence to use Copernicus products: https://cds.climate.copernicus.eu/
   Check the terms of the specific copy the data was read from before redistributing.

3. stations/: {station_license}.

Code that produced this pack is not covered by this file.
"""


def _dataset_card(m: dict, fname: str, variable: str) -> str:
    inits = m["inits"]
    caveat_lines = "\n".join(f"- {c}" for c in m["caveats"]) or "- none recorded"
    return f"""---
license: other
license_name: mixed-see-card
license_link: LICENSE
pretty_name: AI weather forecast downscaling tutorial data ({m['model']}, France)
tags:
- weather
- downscaling
- climate
- tutorial
size_categories:
- n<1K
configs:
- config_name: stations
  data_files: stations/metadata.parquet
- config_name: observations
  data_files: stations/observations.parquet
---

# Downscaling AI weather forecasts to stations: tutorial data pack

Everything needed to follow the tutorial without a GPU or any data account:
an ensemble forecast from **{m['model']}** over France, the model's surface
height, the ERA5 analysis at every forecast valid time, and quality-checked
hourly station observations.

| | |
|---|---|
| Forecast model | [{m['model_repo']}](https://huggingface.co/{m['model_repo']}) |
| Initialisations | {len(inits)}: {inits[0]} ... {inits[-1]} |
| Lead times | {m['lead_hours'][0]}-{m['lead_hours'][-1]} h, {len(m['lead_hours'])} steps |
| Members | {m['members']} |
| Grid | 0.25 deg, lat {m['grid']['lat']}, lon {m['grid']['lon']} |
| Stations | {m['stations']} ({m['holdout_stations']} flagged `holdout`) |
| Observation window | {m['observation_window'][0]} to {m['observation_window'][1]} UTC |

## Files

- `forecasts/{fname}`: `{variable}` with dims (time = init, lead_time, ensemble, lat, lon), plus `orography` (m).
- `analysis/era5_{variable}.zarr`: ERA5 `{variable}` (time, lat, lon) at every forecast valid time.
- `stations/metadata.parquet`: id, name, latitude, longitude, elevation_m, holdout.
- `stations/observations.parquet`: long table, one row per station and hour.

Units: temperatures in K, heights in m, times in UTC. Longitudes are signed (-180..180).

## Load

```python
from weather_pipeline.datapack import load_pack
pack = load_pack("<this-repo-id>")       # or a local path
pack.forecast, pack.orography, pack.stations, pack.analysis
```

## Caveats (read before using any number from this data)

{caveat_lines}
- The ensemble spread comes from the model's own stochastic sampling only: every member starts
  from the same ERA5 state, so initial-condition uncertainty is not represented. Expect
  under-dispersion at short lead times.
- Forecasts are initialised from ERA5, not from the operational analysis the model was
  fine-tuned on.
- Station siting is not controlled (urban, valley, exposed sites all included); QC is limited to a
  range check and >= 90% completeness.
- A single event with few initialisations: results illustrate the method, they do not measure skill.

## Sources and licenses

- Forecasts: produced with {m['model_repo']} (ECMWF), CC-BY-4.0. Attribute ECMWF.
- Analysis and orography: ERA5 (Copernicus Climate Change Service). Contains modified Copernicus
  Climate Change Service information. Check the terms of the copy you redistribute from.
- Station observations: {m['station_license'] or 'see provider'}.

Provenance (package versions, exact inits) is in `manifest.json`.
"""


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
class PackAnalysis:
    """The pack's ERA5 crop behind the Earth2Studio source call signature,
    so analysis_baseline works unchanged: source(times, [variable])."""

    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def __call__(self, times, variables):
        t = pd.DatetimeIndex(times).to_numpy().astype("datetime64[ns]")
        missing = np.setdiff1d(t, self.ds.time.values)
        if missing.size:
            raise KeyError(f"{missing.size} requested time(s) not in the pack, e.g. {missing[:3]}")
        arrs = [self.ds[v].sel(time=t) for v in variables]
        return xr.concat(arrs, dim="variable").assign_coords(variable=list(variables)) \
                 .transpose("time", "variable", "lat", "lon")


@dataclass
class Pack:
    forecast: xr.Dataset
    orography: xr.DataArray
    stations: xr.Dataset
    analysis: PackAnalysis
    manifest: dict
    path: Path

    def init(self, i: int) -> xr.Dataset:
        """One initialisation, as a per-init zarr would look (for per-case scoring)."""
        return self.forecast.isel(time=[i])


def _stations_from_parquet(folder: Path) -> xr.Dataset:
    meta = pd.read_parquet(folder / "metadata.parquet")
    obs = pd.read_parquet(folder / "observations.parquet")
    obs["station_id"] = obs["station_id"].astype(str)
    variables = [c for c in obs.columns if c not in ("station_id", "time")]
    ids = meta["station_id"].astype(str).to_list()
    hours = pd.date_range(obs["time"].min(), obs["time"].max(), freq="1h")
    data = {}
    for v in variables:
        wide = obs.pivot_table(index="time", columns="station_id", values=v, aggfunc="first")
        data[v] = (("time", "station"), wide.reindex(index=hours, columns=ids).to_numpy("f4"))
    ds = xr.Dataset(data, coords={
        "time": hours.to_numpy().astype("datetime64[ns]"),
        "station": np.array(ids, dtype=object),
        "latitude": ("station", meta["latitude"].to_numpy("f8")),
        "longitude": ("station", meta["longitude"].to_numpy("f8")),
        "elevation": ("station", meta["elevation_m"].to_numpy("f8")),
        "name": ("station", meta["name"].astype(str).to_numpy(dtype=object)),
    })
    ds["holdout"] = ("station", meta["holdout"].to_numpy(bool))
    for v in variables:
        ds[v].attrs["units"] = "K"
    return ds


def load_pack(path_or_repo: str | Path, revision: str | None = None) -> Pack:
    """Local folder, or a Hugging Face dataset repo id (downloaded and cached)."""
    p = Path(path_or_repo)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(repo_id=str(path_or_repo), repo_type="dataset", revision=revision))
    manifest = json.loads((p / "manifest.json").read_text())
    var, model = manifest["variable"], manifest["model"]
    fc = xr.open_zarr(p / "forecasts" / f"{model}_{var}.zarr")
    oro = fc["orography"].load()
    fc = fc.drop_vars("orography")
    analysis = PackAnalysis(xr.open_zarr(p / "analysis" / f"era5_{var}.zarr").load())
    return Pack(fc, oro, _stations_from_parquet(p / "stations"), analysis, manifest, p)