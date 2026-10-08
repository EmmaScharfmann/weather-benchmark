"""
Download CERRA 2 m temperature over France and store it as zarr.

Setup, in $HOME/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <API-KEY>

Usage:
    uv run python scripts/download_cerra.py --years 2014 2020 --out data/cerra
    uv run python scripts/download_cerra.py --years 2014 2020 --out data/cerra --consolidate

What it does:
  * static fields (orography, land-sea mask) once, from one analysis time
  * one CDS request per month: 2 m temperature at 00/06/12/18 UTC
  * every file is cropped to France with one fixed index box (saved in
    crop.json), so all months share exactly the same grid
  * each month is its own zarr (data/cerra/months/t2m_YYYYMM.zarr): an
    interrupted run is resumed by rerunning the same command
  * checks per month: no missing time steps, values plausible in K
  * --consolidate merges the months into data/cerra/cerra_france_t2m.zarr
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import shutil
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DATASET = "reanalysis-cerra-single-levels"
FRANCE_BOX = (41.0, 51.5, -5.5, 10.0)          # lat_min, lat_max, lon_min, lon_max
TIMES = ["00:00", "06:00", "12:00", "18:00"]   # UTC analysis times
G = 9.80665                                    # standard gravity, m s-2
T_RANGE_K = (220.0, 330.0)                     # plausible 2 m temperature, exclusive bounds

STATIC_VARIABLES = ["orography", "land_sea_mask"]
T2M_VARIABLE = "2m_temperature"
OROGRAPHY_NAMES = ("orog", "z", "orography", "h")   # possible cfgrib short names
LSM_NAMES = ("lsm", "land_sea_mask")
T2M_NAMES = ("t2m", "2t")
DROP_COORDS = ("step", "heightAboveGround", "surface", "valid_time", "number")
GEOPOTENTIAL_UNITS = ("m**2s**-2", "m2s-2")
GEOPOTENTIAL_MIN = 20000.0     # a max above this cannot be a height in m: it is geopotential

CROP_FILE = "crop.json"
GRIB_DIR = "grib"
MONTHS_DIR = "months"
STATIC_ZARR = "cerra_france_static.zarr"
T2M_ZARR = "cerra_france_t2m.zarr"
SANITY_PNG = "sanity.png"
CONSOLIDATED_TIME_CHUNK = 124  # 31 days x 4 steps: about one month per chunk
PLOT_REF_LAT = 46.0            # latitude used for the map aspect ratio

DEFAULT_YEARS = [2014, 2020]
DEFAULT_OUT = "data/cerra"
DEFAULT_WORKERS = 2

Opener = Callable[[Path], xr.Dataset]

logger = logging.getLogger("cerra")


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------
def build_request(variables: Sequence[str], year: int, month: int,
                  days: Sequence[int] | None = None, times: Sequence[str] = TIMES) -> dict:
    """Build a CDS request for CERRA single-level analyses.

    Parameters
    ----------
    variables : Sequence[str]
        CDS variable names, e.g. ``["2m_temperature"]``.
    year, month : int
        Requested month.
    days : Sequence[int], optional
        Days of the month. Defaults to all of them.
    times : Sequence[str]
        UTC times as ``"HH:MM"``.

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


def crop_box(lat2d: np.ndarray, lon2d: np.ndarray,
             box: tuple[float, float, float, float] = FRANCE_BOX) -> dict:
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
def expected_times(year: int, month: int, times: Sequence[str] = TIMES) -> pd.DatetimeIndex:
    """All expected time stamps of a month: every day at each of ``times``."""
    ndays = calendar.monthrange(year, month)[1]
    days = pd.date_range(f"{year}-{month:02d}-01", periods=ndays, freq="D")
    return pd.DatetimeIndex([d + pd.Timedelta(t + ":00") for d in days for t in times])


def check_month(ds: xr.Dataset, year: int, month: int) -> list[str]:
    """Check completeness and plausibility of one month of ``t2m``.

    Parameters
    ----------
    ds : xr.Dataset
        Month dataset with a ``t2m`` variable in K and a ``time`` dim.
    year, month : int
        Month the dataset should cover.

    Returns
    -------
    list[str]
        Human-readable problems (missing or duplicate times, NaNs, values
        outside ``T_RANGE_K``). Empty if the month is fine.
    """
    problems = []
    got = pd.DatetimeIndex(ds["time"].values)
    missing = expected_times(year, month).difference(got)
    if len(missing):
        problems.append(f"{len(missing)} missing time(s), e.g. {list(missing[:3])}")
    if got.has_duplicates:
        problems.append("duplicate times")
    t = ds["t2m"].values
    if np.isnan(t).any():
        problems.append(f"{int(np.isnan(t).sum())} NaN value(s)")
    lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
    if not (T_RANGE_K[0] < lo and hi < T_RANGE_K[1]):
        problems.append(f"t2m range {lo:.1f}..{hi:.1f} K outside {T_RANGE_K}")
    return problems


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def load_or_make_crop(out: Path, ds_std: xr.Dataset) -> dict:
    """Read ``crop.json`` from ``out``, or compute it from ``ds_std`` and save it.

    Parameters
    ----------
    out : Path
        Output directory.
    ds_std : xr.Dataset
        Standardised full-grid dataset (only used if no crop is saved yet).

    Returns
    -------
    dict
        Crop as returned by ``crop_box``.
    """
    f = out / CROP_FILE
    if f.exists():
        return json.loads(f.read_text())
    crop = crop_box(ds_std["lat"].values, ds_std["lon"].values)
    f.write_text(json.dumps(crop, indent=2))
    logger.info("crop fixed: y %d:%d, x %d:%d of grid %s -> %d x %d points",
                crop["y0"], crop["y1"], crop["x0"], crop["x1"], crop["grid_shape"],
                crop["y1"] - crop["y0"], crop["x1"] - crop["x0"])
    return crop


def do_static(out: Path, year: int, client: Any | None = None, opener: Opener = open_grib) -> Path:
    """Download, crop and save orography (m) and land-sea mask. Skipped if already done.

    Parameters
    ----------
    out : Path
        Output directory.
    year : int
        Year of the analysis time the fields are taken from (1 Jan, 00 UTC).
    client : cdsapi.Client, optional
        CDS client, passed to ``retrieve``.
    opener : Callable[[Path], xr.Dataset]
        GRIB reader (replaceable in tests).

    Returns
    -------
    Path
        Path of ``cerra_france_static.zarr``.

    Raises
    ------
    KeyError
        If orography or land-sea mask is missing from the GRIB.
    """
    target = out / STATIC_ZARR
    if target.exists():
        logger.info("static fields already done")
        return target
    grib = out / GRIB_DIR / "static.grib"
    if not grib.exists():
        retrieve(build_request(STATIC_VARIABLES, year, 1, days=[1], times=["00:00"]), grib, client)
    ds = standardise(opener(grib))
    crop = load_or_make_crop(out, ds)
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


def do_month(out: Path, year: int, month: int, keep_grib: bool = False,
             client: Any | None = None, opener: Opener = open_grib) -> tuple[str, list[str]]:
    """Download, crop, check and save one month of 2 m temperature.

    Skipped if the month's zarr already exists. If the checks fail, nothing is
    written and the GRIB is kept for inspection; it is reused (not re-downloaded)
    on the next run until deleted by hand.

    Parameters
    ----------
    out : Path
        Output directory.
    year, month : int
        Month to process.
    keep_grib : bool
        Keep the GRIB after a successful write.
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
    target = out / MONTHS_DIR / f"t2m_{tag}.zarr"
    if target.exists():
        return tag, []
    grib = out / GRIB_DIR / f"t2m_{tag}.grib"
    if not grib.exists():
        logger.info("requesting %s", tag)
        retrieve(build_request([T2M_VARIABLE], year, month), grib, client)
    ds = standardise(opener(grib))
    crop = load_or_make_crop(out, ds)
    ds = apply_crop(ds, crop)
    var = next((v for v in ds.data_vars if v in T2M_NAMES), None)
    if var is None:
        raise KeyError(f"2 m temperature not found; variables: {list(ds.data_vars)}")
    ds = ds[[var]].rename({var: "t2m"}).sortby("time")
    ds["t2m"] = ds["t2m"].astype("f4").assign_attrs(units="K")
    problems = check_month(ds, year, month)
    if problems:
        return tag, problems
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    _clean(ds.load()).to_zarr(tmp, mode="w")
    tmp.rename(target)
    if not keep_grib:
        grib.unlink()
    return tag, []


def consolidate(out: Path, years: tuple[int, int]) -> Path:
    """Concatenate all monthly zarrs of ``years`` into ``cerra_france_t2m.zarr``.

    Parameters
    ----------
    out : Path
        Output directory.
    years : tuple[int, int]
        First and last year, inclusive.

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
    tags = [f"{y}{m:02d}" for y in range(years[0], years[1] + 1) for m in range(1, 13)]
    paths = [out / MONTHS_DIR / f"t2m_{t}.zarr" for t in tags]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month(s) not downloaded yet, e.g. {missing[:3]}")
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    ds = ds.chunk({"time": CONSOLIDATED_TIME_CHUNK, "y": -1, "x": -1})
    times = pd.DatetimeIndex(ds.time.values)
    assert times.is_monotonic_increasing and not times.has_duplicates
    st = xr.open_zarr(out / STATIC_ZARR)
    if not (np.array_equal(st.lat.values, ds.lat.values) and np.array_equal(st.lon.values, ds.lon.values)):
        raise ValueError("static grid and t2m grid differ")
    ds.attrs = {"source": "CERRA (C3S), 2 m temperature analysis, cropped to France",
                "years": f"{years[0]}-{years[1]}", "times_utc": ",".join(TIMES),
                "license": "Copernicus licence; contains modified C3S information"}
    target = out / T2M_ZARR
    _clean(ds).to_zarr(target, mode="w")
    logger.info("consolidated %d time steps, grid %d x %d -> %s",
                ds.sizes["time"], ds.sizes["y"], ds.sizes["x"], target)
    return target


def sanity_plot(out: Path, tag: str, path: Path) -> None:
    """Save a figure of the month's warmest time step next to the orography.

    Parameters
    ----------
    out : Path
        Output directory (must contain the month and the static zarr).
    tag : str
        Month as ``"YYYYMM"``.
    path : Path
        PNG file to write.
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
        a_.set_aspect(1 / np.cos(np.deg2rad(PLOT_REF_LAT)))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("sanity plot: %s", path)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

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
    p.add_argument("--years", nargs=2, type=int, default=DEFAULT_YEARS, metavar=("FIRST", "LAST"))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent CDS requests (keep small)")
    p.add_argument("--keep-grib", action="store_true")
    p.add_argument("--consolidate", action="store_true", help="merge finished months into one zarr")
    p.add_argument("--dry-run", action="store_true", help="print the requests, download nothing")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    years = (a.years[0], a.years[1])

    if a.consolidate:
        consolidate(out, years)
        return 0
    months = [(y, m) for y in range(years[0], years[1] + 1) for m in range(1, 13)]
    if a.dry_run:
        print(json.dumps(build_request(STATIC_VARIABLES, years[0], 1, days=[1], times=["00:00"]), indent=1))
        print(json.dumps(build_request([T2M_VARIABLE], *months[0]), indent=1))
        print(f"... {len(months)} monthly requests in total")
        return 0

    do_static(out, years[0])
    failed: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(do_month, out, y, m, a.keep_grib): (y, m) for y, m in months}
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

    done = sorted(p.stem.split("_")[1] for p in (out / MONTHS_DIR).glob("t2m_*.zarr"))
    if done:
        sanity_plot(out, done[len(done) // 2 if len(done) > 1 else 0], out / SANITY_PNG)
    print(f"\n{len(done)}/{len(months)} months done; {len(failed)} with problems")
    for tag, probs in sorted(failed.items()):
        print(f"  {tag}: {'; '.join(probs)}")
    print("rerun the same command to retry; then add --consolidate" if failed or len(done) < len(months)
          else "all months done: now run with --consolidate")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())