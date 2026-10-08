"""
Download CERRA 2 m temperature over France and store it as zarr.

Setup, in $HOME/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <API-KEY>

Usage (settings from configs/download_cerra.yaml, overridable on the command line):
    uv run python scripts/download_cerra.py
    uv run python scripts/download_cerra.py --years 2014 2020 --out data/cerra
    uv run python scripts/download_cerra.py --consolidate
    uv run python scripts/download_cerra.py --config my_run.yaml

What it does:
  * static fields (orography, land-sea mask) once, from one analysis time
  * one CDS request per month: 2 m temperature at the configured UTC times
  * every file is cropped with one fixed index box (saved in crop.json), so
    all months share exactly the same grid
  * each month is its own zarr (<out>/months/t2m_YYYYMM.zarr): an
    interrupted run is resumed by rerunning the same command
  * checks per month: no missing time steps, values plausible in K
  * --consolidate merges the months into <out>/cerra_france_t2m.zarr
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from cerra_constants import (CROP_FILE, DATASET, DROP_COORDS, GEOPOTENTIAL_MIN, GEOPOTENTIAL_UNITS, G,
                             GRIB_DIR, LSM_NAMES, MONTHS_DIR, OROGRAPHY_NAMES, SANITY_PNG, STATIC_VARIABLES,
                             STATIC_ZARR, T2M_NAMES, T2M_VARIABLE, T2M_ZARR)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "download_cerra.yaml"

Opener = Callable[[Path], xr.Dataset]

logger = logging.getLogger("cerra")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class DownloadConfig:
    """Run settings, read from YAML (see configs/download_cerra.yaml).

    Attributes
    ----------
    years : tuple[int, int]
        First and last year, inclusive.
    out : Path
        Output directory.
    workers : int
        Concurrent CDS requests.
    keep_grib : bool
        Keep GRIB files after a successful write.
    box : tuple[float, float, float, float]
        Crop region ``(lat_min, lat_max, lon_min, lon_max)``, degrees. Only used
        until ``crop.json`` exists.
    times : list[str]
        UTC analysis times as ``"HH:MM"``.
    t_range_k : tuple[float, float]
        Plausible 2 m temperature range, K, exclusive bounds.
    time_chunk : int
        Chunk length along time in the consolidated zarr.
    plot_ref_lat : float
        Latitude used for the sanity-plot aspect ratio, degrees.
    """

    years: tuple[int, int]
    out: Path
    workers: int
    keep_grib: bool
    box: tuple[float, float, float, float]
    times: list[str]
    t_range_k: tuple[float, float]
    time_chunk: int
    plot_ref_lat: float

    def __post_init__(self):
        if not all(isinstance(t, str) and re.fullmatch(r"\d{2}:\d{2}", t) for t in self.times):
            raise ValueError(f"times must be quoted 'HH:MM' strings, got {self.times}")
        if len(self.years) != 2 or len(self.box) != 4 or len(self.t_range_k) != 2:
            raise ValueError("years and t_range_k need 2 values, box needs 4")
        self.years = (int(self.years[0]), int(self.years[1]))
        self.out = Path(self.out)
        self.box = tuple(float(v) for v in self.box)
        self.times = list(self.times)
        self.t_range_k = (float(self.t_range_k[0]), float(self.t_range_k[1]))


def load_config(path: str | Path) -> DownloadConfig:
    """Read a YAML run configuration.

    Parameters
    ----------
    path : str or Path
        YAML file whose keys are the ``DownloadConfig`` fields.

    Returns
    -------
    DownloadConfig
        Validated settings.

    Raises
    ------
    TypeError
        If a key is missing or unknown.
    ValueError
        If a value has the wrong shape or format.
    """
    return DownloadConfig(**yaml.safe_load(Path(path).read_text()))


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def build_request(variables: Sequence[str], year: int, month: int, times: Sequence[str],
                  days: Sequence[int] | None = None) -> dict:
    """Build a CDS request for CERRA single-level analyses.

    Parameters
    ----------
    variables : Sequence[str]
        CDS variable names, e.g. ``["2m_temperature"]``.
    year, month : int
        Requested month.
    times : Sequence[str]
        UTC times as ``"HH:MM"``.
    days : Sequence[int], optional
        Days of the month. Defaults to all of them.

    Returns
    -------
    dict
        Request body for ``cdsapi.Client.retrieve`` (GRIB output).
    """
    ndays = calendar.monthrange(year, month)[1]
    return {
        "level_type": "surface_or_atmosphere",
        "data_type": "reanalysis",
        "product_type": "analysis",
        "variable": list(variables),
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in (days or range(1, ndays + 1))],
        "time": list(times),
        "data_format": "grib",
    }


def retrieve(request: dict, target: Path, client: Any | None = None) -> Path:
    """Download one CDS request to a GRIB file.

    The file is written to ``<target>.part`` and renamed when complete, so an
    interrupted download never looks finished.

    Parameters
    ----------
    request : dict
        Output of ``build_request``.
    target : Path
        Final GRIB path.
    client : cdsapi.Client, optional
        CDS client. A new one is created if None.

    Returns
    -------
    Path
        ``target``.
    """
    if client is None:
        import cdsapi
        client = cdsapi.Client()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    client.retrieve(DATASET, request, str(part))
    part.rename(target)
    return target


# ---------------------------------------------------------------------------
# GRIB -> standard dataset
# ---------------------------------------------------------------------------
def open_grib(path: Path) -> xr.Dataset:
    """Open a GRIB file with cfgrib, without writing an index file next to it."""
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def standardise(ds: xr.Dataset) -> xr.Dataset:
    """Convert cfgrib's CERRA layout to a standard form.

    Uses the valid time as ``time`` (analyses have step 0), drops scalar
    coordinates, renames latitude/longitude to ``lat``/``lon`` and wraps
    longitudes to [-180, 180).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset as opened by ``open_grib``.

    Returns
    -------
    xr.Dataset
        Dataset with 2-D ``lat``/``lon`` coordinates on the native grid dims.

    Raises
    ------
    ValueError
        If lat/lon are not 2-D (i.e. not the Lambert grid).
    """
    if "valid_time" in ds.coords and "time" in ds.dims:
        ds = ds.assign_coords(time=ds["valid_time"].values) if ds["valid_time"].ndim == 1 else ds
    for c in DROP_COORDS:
        if c in ds.coords and c not in ds.dims:
            ds = ds.drop_vars(c)
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords}
    ds = ds.rename(ren)
    if ds["lat"].ndim != 2:
        raise ValueError(f"expected 2-D lat/lon (Lambert grid), got dims {ds['lat'].dims}")
    return ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0)


def crop_box(lat2d: np.ndarray, lon2d: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    """Smallest index rectangle containing every grid point inside a lat/lon box.

    A rectangle in index space keeps the array regular for a CNN. Its corners
    extend slightly outside the box, which is harmless.

    Parameters
    ----------
    lat2d, lon2d : np.ndarray
        Full-grid coordinates, shape (ny, nx), degrees.
    box : tuple[float, float, float, float]
        ``(lat_min, lat_max, lon_min, lon_max)``.

    Returns
    -------
    dict
        ``{"y0", "y1", "x0", "x1"}`` (end-exclusive indices), ``"grid_shape"``
        ``[ny, nx]`` and ``"box"``. JSON-serialisable.

    Raises
    ------
    ValueError
        If no grid point falls inside the box.
    """
    la0, la1, lo0, lo1 = box
    inside = (lat2d >= la0) & (lat2d <= la1) & (lon2d >= lo0) & (lon2d <= lo1)
    if not inside.any():
        raise ValueError(f"no CERRA grid point inside {box}")
    ys, xs = np.nonzero(inside)
    return {"y0": int(ys.min()), "y1": int(ys.max()) + 1, "x0": int(xs.min()), "x1": int(xs.max()) + 1,
            "grid_shape": [int(s) for s in lat2d.shape], "box": list(box)}


def apply_crop(ds: xr.Dataset, crop: dict) -> xr.Dataset:
    """Apply a ``crop_box`` result and rename the grid dims to ``(y, x)``.

    Parameters
    ----------
    ds : xr.Dataset
        Standardised dataset on the full CERRA grid.
    crop : dict
        Output of ``crop_box``.

    Returns
    -------
    xr.Dataset
        Cropped dataset with dims ``(..., y, x)``.

    Raises
    ------
    ValueError
        If the dataset's grid shape differs from the one the crop was made on.
    """
    ydim, xdim = ds["lat"].dims
    shape = [int(ds.sizes[ydim]), int(ds.sizes[xdim])]
    if shape != crop["grid_shape"]:
        raise ValueError(f"grid {shape} differs from the saved crop's grid {crop['grid_shape']}")
    out = ds.isel({ydim: slice(crop["y0"], crop["y1"]), xdim: slice(crop["x0"], crop["x1"])})
    return out.rename({ydim: "y", xdim: "x"}) if (ydim, xdim) != ("y", "x") else out


def _clean(ds: xr.Dataset) -> xr.Dataset:
    """Drop GRIB encodings and attributes in place so the dataset writes cleanly to zarr."""
    for v in ds.variables.values():
        v.encoding.clear()
    ds.attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB")}
    return ds


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def expected_times(year: int, month: int, times: Sequence[str]) -> pd.DatetimeIndex:
    """All expected time stamps of a month: every day at each of ``times`` (UTC, "HH:MM")."""
    ndays = calendar.monthrange(year, month)[1]
    days = pd.date_range(f"{year}-{month:02d}-01", periods=ndays, freq="D")
    return pd.DatetimeIndex([d + pd.Timedelta(t + ":00") for d in days for t in times])


def check_month(ds: xr.Dataset, year: int, month: int, times: Sequence[str],
                t_range_k: tuple[float, float]) -> list[str]:
    """Check completeness and plausibility of one month of ``t2m``.

    Parameters
    ----------
    ds : xr.Dataset
        Month dataset with a ``t2m`` variable in K and a ``time`` dim.
    year, month : int
        Month the dataset should cover.
    times : Sequence[str]
        Expected UTC times per day, "HH:MM".
    t_range_k : tuple[float, float]
        Plausible range, K, exclusive bounds.

    Returns
    -------
    list[str]
        Human-readable problems (missing or duplicate times, NaNs, values out of
        range). Empty if the month is fine.
    """
    problems = []
    got = pd.DatetimeIndex(ds["time"].values)
    missing = expected_times(year, month, times).difference(got)
    if len(missing):
        problems.append(f"{len(missing)} missing time(s), e.g. {list(missing[:3])}")
    if got.has_duplicates:
        problems.append("duplicate times")
    t = ds["t2m"].values
    if np.isnan(t).any():
        problems.append(f"{int(np.isnan(t).sum())} NaN value(s)")
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    if not (t_range_k[0] < lo and hi < t_range_k[1]):
        problems.append(f"t2m range {lo:.1f}..{hi:.1f} K outside {t_range_k}")
    return problems


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def load_or_make_crop(out: Path, ds_std: xr.Dataset, box: tuple[float, float, float, float]) -> dict:
    """Read ``crop.json`` from ``out``, or compute it from ``ds_std`` and ``box`` and save it.

    Parameters
    ----------
    out : Path
        Output directory.
    ds_std : xr.Dataset
        Standardised full-grid dataset (only used if no crop is saved yet).
    box : tuple[float, float, float, float]
        Crop region (only used if no crop is saved yet).

    Returns
    -------
    dict
        Crop as returned by ``crop_box``.
    """
    f = out / CROP_FILE
    if f.exists():
        return json.loads(f.read_text())
    crop = crop_box(ds_std["lat"].values, ds_std["lon"].values, box)
    f.write_text(json.dumps(crop, indent=2))
    logger.info("crop fixed: y %d:%d, x %d:%d of grid %s -> %d x %d points",
                crop["y0"], crop["y1"], crop["x0"], crop["x1"], crop["grid_shape"],
                crop["y1"] - crop["y0"], crop["x1"] - crop["x0"])
    return crop


def do_static(cfg: DownloadConfig, client: Any | None = None, opener: Opener = open_grib) -> Path:
    """Download, crop and save orography (m) and land-sea mask. Skipped if already done.

    The fields are taken from 1 January of the first year, at the first configured time.

    Parameters
    ----------
    cfg : DownloadConfig
        Run settings.
    client : cdsapi.Client, optional
        CDS client, passed to ``retrieve``.
    opener : Callable[[Path], xr.Dataset]
        GRIB reader (replaceable in tests).

    Returns
    -------
    Path
        Path of the static zarr.

    Raises
    ------
    KeyError
        If orography or land-sea mask is missing from the GRIB.
    """
    target = cfg.out / STATIC_ZARR
    if target.exists():
        logger.info("static fields already done")
        return target
    grib = cfg.out / GRIB_DIR / "static.grib"
    if not grib.exists():
        retrieve(build_request(STATIC_VARIABLES, cfg.years[0], 1, cfg.times[:1], days=[1]), grib, client)
    ds = standardise(opener(grib))
    crop = load_or_make_crop(cfg.out, ds, cfg.box)
    ds = apply_crop(ds, crop)
    ds = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
    oro = next((v for v in ds.data_vars if v in OROGRAPHY_NAMES), None)
    lsm = next((v for v in ds.data_vars if v in LSM_NAMES), None)
    if oro is None or lsm is None:
        raise KeyError(f"orography / land-sea mask not found in static GRIB; variables: {list(ds.data_vars)}")
    h = ds[oro].astype("f4")
    if str(h.attrs.get("units", "")).replace(" ", "") in GEOPOTENTIAL_UNITS or float(h.max()) > GEOPOTENTIAL_MIN:
        h = h / G
    st = xr.Dataset({"orography": h.assign_attrs(units="m"),
                     "land_sea_mask": ds[lsm].astype("f4").assign_attrs(units="1")},
                    coords={"lat": ds["lat"], "lon": ds["lon"]})
    st.attrs = {"source": "CERRA (C3S), cropped to France", "crop": json.dumps(crop)}
    _clean(st).to_zarr(target, mode="w")
    logger.info("static: orography %.0f..%.0f m, land fraction %.2f",
                float(st.orography.min()), float(st.orography.max()), float(st.land_sea_mask.mean()))
    return target


def do_month(cfg: DownloadConfig, year: int, month: int,
             client: Any | None = None, opener: Opener = open_grib) -> tuple[str, list[str]]:
    """Download, crop, check and save one month of 2 m temperature.

    Skipped if the month's zarr already exists. If the checks fail, nothing is
    written and the GRIB is kept for inspection; it is reused (not re-downloaded)
    on the next run until deleted by hand.

    Parameters
    ----------
    cfg : DownloadConfig
        Run settings.
    year, month : int
        Month to process.
    client : cdsapi.Client, optional
        CDS client, passed to ``retrieve``.
    opener : Callable[[Path], xr.Dataset]
        GRIB reader (replaceable in tests).

    Returns
    -------
    tuple[str, list[str]]
        ``("YYYYMM", problems)``, with ``problems`` empty on success.

    Raises
    ------
    KeyError
        If 2 m temperature is missing from the GRIB.
    """
    tag = f"{year}{month:02d}"
    target = cfg.out / MONTHS_DIR / f"t2m_{tag}.zarr"
    if target.exists():
        return tag, []
    grib = cfg.out / GRIB_DIR / f"t2m_{tag}.grib"
    if not grib.exists():
        logger.info("requesting %s", tag)
        retrieve(build_request([T2M_VARIABLE], year, month, cfg.times), grib, client)
    ds = standardise(opener(grib))
    crop = load_or_make_crop(cfg.out, ds, cfg.box)
    ds = apply_crop(ds, crop)
    var = next((v for v in ds.data_vars if v in T2M_NAMES), None)
    if var is None:
        raise KeyError(f"2 m temperature not found; variables: {list(ds.data_vars)}")
    ds = ds[[var]].rename({var: "t2m"}).sortby("time")
    ds["t2m"] = ds["t2m"].astype("f4").assign_attrs(units="K")
    problems = check_month(ds, year, month, cfg.times, cfg.t_range_k)
    if problems:
        return tag, problems
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    _clean(ds.load()).to_zarr(tmp, mode="w")
    tmp.rename(target)
    if not cfg.keep_grib:
        grib.unlink()
    return tag, []


def consolidate(cfg: DownloadConfig) -> Path:
    """Concatenate all monthly zarrs of ``cfg.years`` into one zarr.

    Parameters
    ----------
    cfg : DownloadConfig
        Run settings.

    Returns
    -------
    Path
        Path of the consolidated zarr, ``t2m`` with dims (time, y, x).

    Raises
    ------
    FileNotFoundError
        If any month of the period is missing.
    ValueError
        If the t2m grid differs from the static grid.
    """
    y0, y1 = cfg.years
    tags = [f"{y}{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]
    paths = [cfg.out / MONTHS_DIR / f"t2m_{t}.zarr" for t in tags]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month(s) not downloaded yet, e.g. {missing[:3]}")
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    ds = ds.chunk({"time": cfg.time_chunk, "y": -1, "x": -1})
    times = pd.DatetimeIndex(ds.time.values)
    assert times.is_monotonic_increasing and not times.has_duplicates
    st = xr.open_zarr(cfg.out / STATIC_ZARR)
    if not (np.array_equal(st.lat.values, ds.lat.values) and np.array_equal(st.lon.values, ds.lon.values)):
        raise ValueError("static grid and t2m grid differ")
    ds.attrs = {"source": "CERRA (C3S), 2 m temperature analysis, cropped to France",
                "years": f"{y0}-{y1}", "times_utc": ",".join(cfg.times),
                "license": "Copernicus licence; contains modified C3S information"}
    target = cfg.out / T2M_ZARR
    _clean(ds).to_zarr(target, mode="w")
    logger.info("consolidated %d time steps, grid %d x %d -> %s",
                ds.sizes["time"], ds.sizes["y"], ds.sizes["x"], target)
    return target


def sanity_plot(out: Path, tag: str, path: Path, ref_lat: float) -> None:
    """Save a figure of the month's warmest time step next to the orography.

    Parameters
    ----------
    out : Path
        Output directory (must contain the month and the static zarr).
    tag : str
        Month as ``"YYYYMM"``.
    path : Path
        PNG file to write.
    ref_lat : float
        Latitude used for the map aspect ratio, degrees.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = xr.open_zarr(out / MONTHS_DIR / f"t2m_{tag}.zarr")
    st = xr.open_zarr(out / STATIC_ZARR)
    i = int(np.argmax(ds.t2m.mean(("y", "x")).values))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    a = ax[0].pcolormesh(ds.lon, ds.lat, ds.t2m.isel(time=i) - 273.15, cmap="RdYlBu_r", shading="auto")
    fig.colorbar(a, ax=ax[0], label="°C")
    ax[0].set_title(f"CERRA 2 m temperature, {pd.Timestamp(ds.time.values[i]):%Y-%m-%d %H UTC}")
    b = ax[1].pcolormesh(st.lon, st.lat, st.orography, cmap="terrain", shading="auto")
    fig.colorbar(b, ax=ax[1], label="m")
    ax[1].set_title("CERRA orography")
    for a_ in ax:
        a_.set_aspect(1 / np.cos(np.deg2rad(ref_lat)))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("sanity plot: %s", path)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Settings are read from ``--config``; ``--years``, ``--out``, ``--workers``
    and ``--keep-grib`` override the file when given.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 if every month succeeded (or for --consolidate/--dry-run), else 1.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML run settings")
    p.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    p.add_argument("--out")
    p.add_argument("--workers", type=int, help="concurrent CDS requests (keep small)")
    p.add_argument("--keep-grib", action="store_true", default=None)
    p.add_argument("--consolidate", action="store_true", help="merge finished months into one zarr")
    p.add_argument("--dry-run", action="store_true", help="print the requests, download nothing")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    overrides = {"years": a.years, "out": a.out, "workers": a.workers, "keep_grib": a.keep_grib}
    cfg = replace(load_config(a.config), **{k: v for k, v in overrides.items() if v is not None})
    cfg.out.mkdir(parents=True, exist_ok=True)

    if a.consolidate:
        consolidate(cfg)
        return 0
    months = [(y, m) for y in range(cfg.years[0], cfg.years[1] + 1) for m in range(1, 13)]
    if a.dry_run:
        print(json.dumps(build_request(STATIC_VARIABLES, cfg.years[0], 1, cfg.times[:1], days=[1]), indent=1))
        print(json.dumps(build_request([T2M_VARIABLE], *months[0], cfg.times), indent=1))
        print(f"... {len(months)} monthly requests in total")
        return 0

    do_static(cfg)
    failed: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {pool.submit(do_month, cfg, y, m): (y, m) for y, m in months}
        for n, f in enumerate(as_completed(futs), 1):
            y, m = futs[f]
            try:
                tag, problems = f.result()
                if problems:
                    failed[tag] = problems
                logger.info("[%d/%d] %d-%02d %s", n, len(months), y, m,
                            "PROBLEMS: " + "; ".join(problems) if problems else "ok")
            except Exception as e:  # noqa: BLE001
                failed[f"{y}{m:02d}"] = [repr(e)]
                logger.error("[%d/%d] %d-%02d failed: %r", n, len(months), y, m, e)

    done = sorted(p.stem.split("_")[1] for p in (cfg.out / MONTHS_DIR).glob("t2m_*.zarr"))
    if done:
        sanity_plot(cfg.out, done[len(done) // 2 if len(done) > 1 else 0], cfg.out / SANITY_PNG, cfg.plot_ref_lat)
    print(f"\n{len(done)}/{len(months)} months done; {len(failed)} with problems")
    for tag, probs in sorted(failed.items()):
        print(f"  {tag}: {'; '.join(probs)}")
    print("rerun the same command to retry; then add --consolidate" if failed or len(done) < len(months)
          else "all months done: now run with --consolidate")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())