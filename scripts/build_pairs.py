"""
Step 2: training pairs for the downscaler -- ERA5 2 m temperature placed on the
CERRA grid (input) next to CERRA 2 m temperature (target), over France.

    uv run python scripts/build_pairs.py --cerra data/cerra --out data/pairs --years 2014 2020
    # rerun any time: finished months are skipped, months whose CERRA file is
    # not downloaded yet are listed as pending
    uv run python scripts/build_pairs.py --cerra data/cerra --out data/pairs --years 2014 2020 --consolidate

Per month (data/pairs/months/pairs_YYYYMM.zarr), dims (time, y, x):
    era5_t2m   ERA5 2 m temperature, bilinearly interpolated onto the CERRA grid  [K]
    cerra_t2m  CERRA 2 m temperature (the target)                                   [K]
Static (data/pairs/static.zarr), dims (y, x):
    cerra_orography, land_sea_mask        fine-scale terrain from CERRA
    era5_orography                        ERA5 surface height on the CERRA grid [m]
--consolidate merges the months into data/pairs/pairs_france.zarr, computes
normalisation statistics on the TRAINING years only (stats.json), and draws
check.png for one heatwave-like time step.

The bilinear ERA5 field is both the model's main input and the "coarse model
alone" baseline that the downscaler has to beat.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline.regions import crop_indices
from weather_pipeline.station_interp import G, GridInterpolator

logger = logging.getLogger("pairs")
MARGIN_DEG = 1.5      # ERA5 crop margin around the CERRA box, so interpolation never hits an edge


# ---------------------------------------------------------------------------
# ERA5 side
# ---------------------------------------------------------------------------
class Era5OnCerra:
    """Fetches ERA5 t2m, crops it to the CERRA box (+ margin) as soon as it
    arrives, and interpolates it onto the CERRA grid points."""

    def __init__(self, source, cerra_lat: np.ndarray, cerra_lon: np.ndarray, batch: int = 28):
        self.source, self.batch = source, batch
        self.shape = cerra_lat.shape
        la, lo = cerra_lat.ravel(), cerra_lon.ravel()
        self.box = (float(la.min()) - MARGIN_DEG, float(la.max()) + MARGIN_DEG,
                    float(lo.min()) - MARGIN_DEG, float(lo.max()) + MARGIN_DEG)
        self.points = (la, lo)
        self._interp = None
        self._idx = None

    def _crop(self, da: xr.DataArray) -> xr.DataArray:
        if self._idx is None:
            self._idx = crop_indices(self.box, da["lat"].values, da["lon"].values)
        return da.isel(lat=self._idx[0], lon=self._idx[1])

    def _to_cerra(self, da: xr.DataArray) -> np.ndarray:
        if self._interp is None:
            self._interp = GridInterpolator.build(da["lat"].values, da["lon"].values, *self.points)
            if not self._interp.inside.all():
                raise ValueError("some CERRA points fall outside the ERA5 crop; increase MARGIN_DEG")
        out = self._interp.apply(da)                       # (..., station)
        return out.values.reshape(*out.shape[:-1], *self.shape)

    def t2m(self, times) -> np.ndarray:
        times = pd.DatetimeIndex(times)
        parts = []
        for i in range(0, times.size, self.batch):
            da = self.source(list(times[i:i + self.batch]), ["t2m"]).sel(variable="t2m", drop=True)
            parts.append(self._to_cerra(self._crop(da)))
        return np.concatenate(parts, axis=0).astype("f4")

    def static_height(self, z_geopotential: xr.DataArray) -> np.ndarray:
        return self._to_cerra(self._crop(z_geopotential / G)).astype("f4")


# ---------------------------------------------------------------------------
# months
# ---------------------------------------------------------------------------
def _clean(ds):
    for v in ds.variables.values():
        v.encoding.clear()
    return ds


def month_checks(era5: np.ndarray, cerra: np.ndarray) -> dict:
    ok = np.isfinite(era5) & np.isfinite(cerra)
    d = (cerra - era5)[ok]
    corr = float(np.corrcoef(era5[ok], cerra[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    return {"nan_era5": int((~np.isfinite(era5)).sum()), "nan_cerra": int((~np.isfinite(cerra)).sum()),
            "corr": round(corr, 4), "mean_diff_K": round(float(d.mean()), 3), "rmse_K": round(float(np.sqrt((d ** 2).mean())), 3)}


def do_month(cerra_dir: Path, out: Path, tag: str, e2c: Era5OnCerra, static_lat, static_lon) -> dict | None:
    target = out / "months" / f"pairs_{tag}.zarr"
    if target.exists():
        return None
    c = xr.open_zarr(cerra_dir / "months" / f"t2m_{tag}.zarr").load()
    if not (np.array_equal(c.lat.values, static_lat) and np.array_equal(c.lon.values, static_lon)):
        raise ValueError(f"{tag}: CERRA month grid differs from the static grid")
    t0 = time.time()
    era5 = e2c.t2m(c.time.values)
    rep = month_checks(era5, c.t2m.values)
    rep.update(tag=tag, steps=int(c.sizes["time"]), fetch_seconds=round(time.time() - t0, 1))
    problems = []
    if rep["nan_era5"] or rep["nan_cerra"]:
        problems.append("NaN values")
    if not rep["corr"] > 0.9:
        problems.append(f"low correlation {rep['corr']}")
    if abs(rep["mean_diff_K"]) > 3:
        problems.append(f"large mean difference {rep['mean_diff_K']} K")
    rep["problems"] = problems
    if problems:
        return rep                                          # nothing written
    ds = xr.Dataset({"era5_t2m": (("time", "y", "x"), era5), "cerra_t2m": (("time", "y", "x"), c.t2m.values.astype("f4"))},
                    coords={"time": c.time.values, "lat": (("y", "x"), static_lat), "lon": (("y", "x"), static_lon)})
    for v in ("era5_t2m", "cerra_t2m"):
        ds[v].attrs["units"] = "K"
    ds["era5_t2m"].attrs["long_name"] = "ERA5 2 m temperature, bilinear onto the CERRA grid"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True)
    _clean(ds).to_zarr(tmp, mode="w"); tmp.rename(target)
    return rep


def do_static(cerra_dir: Path, out: Path, e2c: Era5OnCerra, z_era5: xr.DataArray) -> Path:
    target = out / "static.zarr"
    if target.exists():
        return target
    st = xr.open_zarr(cerra_dir / "cerra_france_static.zarr").load()
    h_era5 = e2c.static_height(z_era5)
    ds = xr.Dataset({"cerra_orography": st.orography.astype("f4"), "land_sea_mask": st.land_sea_mask.astype("f4"),
                     "era5_orography": (("y", "x"), h_era5)},
                    coords={"lat": st.lat, "lon": st.lon})
    for v in ("cerra_orography", "era5_orography"):
        ds[v].attrs["units"] = "m"
    out.mkdir(parents=True, exist_ok=True)
    _clean(ds).to_zarr(target, mode="w")
    dh = ds.cerra_orography - ds.era5_orography
    logger.info("static: CERRA orography %.0f..%.0f m, ERA5 (on CERRA grid) %.0f..%.0f m, difference %.0f..%.0f m",
                float(ds.cerra_orography.min()), float(ds.cerra_orography.max()),
                float(ds.era5_orography.min()), float(ds.era5_orography.max()), float(dh.min()), float(dh.max()))
    return target


# ---------------------------------------------------------------------------
# consolidate: merge, statistics (training years only), checks, figure
# ---------------------------------------------------------------------------
def consolidate(out: Path, years, train_years) -> dict:
    tags = [f"{y}{m:02d}" for y in range(years[0], years[1] + 1) for m in range(1, 13)]
    paths = [out / "months" / f"pairs_{t}.zarr" for t in tags]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month(s) not built yet, e.g. {missing[:3]}")
    ds = xr.open_mfdataset([str(p) for p in paths], engine="zarr", combine="nested", concat_dim="time")
    ds = ds.chunk({"time": 124, "y": -1, "x": -1})
    t = pd.DatetimeIndex(ds.time.values)
    assert t.is_monotonic_increasing and not t.has_duplicates
    st = xr.open_zarr(out / "static.zarr").load()

    train = ds.sel(time=slice(f"{train_years[0]}-01-01", f"{train_years[1]}-12-31T23:59"))
    resid = train.cerra_t2m - train.era5_t2m
    dh = st.cerra_orography - st.era5_orography
    stats = {
        "train_years": list(train_years), "val_years": [y for y in range(years[0], years[1] + 1) if y > train_years[1]],
        "n_train_steps": int(train.sizes["time"]),
        "era5_t2m": {"mean": float(train.era5_t2m.mean().compute()), "std": float(train.era5_t2m.std().compute())},
        "residual": {"mean": float(resid.mean().compute()), "std": float(resid.std().compute())},
        "cerra_orography": {"mean": float(st.cerra_orography.mean()), "std": float(st.cerra_orography.std())},
        "orography_diff": {"mean": float(dh.mean()), "std": float(dh.std())},
    }
    # physical sanity: the time-mean residual should fall with height above the ERA5 surface
    mean_resid = resid.mean("time").compute().values.ravel()
    x = dh.values.ravel()
    ok = np.isfinite(mean_resid) & np.isfinite(x)
    slope = float(np.polyfit(x[ok], mean_resid[ok], 1)[0])
    stats["checks"] = {"residual_vs_height_slope_K_per_km": round(slope * 1000, 2),
                       "baseline_rmse_train_K": round(float(np.sqrt((resid ** 2).mean().compute())), 3)}
    ds.attrs = {"source": "ERA5 (bilinear onto CERRA grid) and CERRA, 2 m temperature, France",
                "years": f"{years[0]}-{years[1]}", "train_years": f"{train_years[0]}-{train_years[1]}"}
    _clean(ds).to_zarr(out / "pairs_france.zarr", mode="w")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("consolidated %d steps; residual mean %.2f K, std %.2f K; slope vs height %.2f K/km",
                ds.sizes["time"], stats["residual"]["mean"], stats["residual"]["std"], slope * 1000)
    _figure(out)
    return stats


def _figure(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds, st = xr.open_zarr(out / "pairs_france.zarr"), xr.open_zarr(out / "static.zarr")
    afternoons = ds.time.dt.hour == 12
    i = int(ds.cerra_t2m.where(afternoons).mean(("y", "x")).argmax().compute())   # hottest midday
    e, c = ds.era5_t2m.isel(time=i).load() - 273.15, ds.cerra_t2m.isel(time=i).load() - 273.15
    lo, hi = float(np.nanpercentile(c, 1)), float(np.nanpercentile(c, 99))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    kw = dict(shading="auto")
    for a, f, title in ((ax[0], e, "ERA5, bilinear onto CERRA grid"), (ax[1], c, "CERRA (target)")):
        m = a.pcolormesh(st.lon, st.lat, f, cmap="RdYlBu_r", vmin=lo, vmax=hi, **kw); a.set_title(title)
        fig.colorbar(m, ax=a, label="°C")
    d = c - e; lim = float(np.nanpercentile(np.abs(d), 99))
    m = ax[2].pcolormesh(st.lon, st.lat, d, cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)
    ax[2].set_title("CERRA - ERA5: what the downscaler must learn"); fig.colorbar(m, ax=ax[2], label="K")
    dh = (st.cerra_orography - st.era5_orography).values.ravel()
    ax[3].scatter(dh, d.values.ravel(), s=1, alpha=.2)
    ax[3].plot([dh.min(), dh.max()], [-0.0065 * dh.min(), -0.0065 * dh.max()], "k--", lw=1, label="−6.5 K/km")
    ax[3].set_xlabel("CERRA − ERA5 surface height (m)"); ax[3].set_ylabel("CERRA − ERA5 temperature (K)")
    ax[3].set_title("Difference vs height difference"); ax[3].legend()
    for a in ax[:3]:
        a.set_aspect(1 / np.cos(np.deg2rad(46)))
    fig.suptitle(f"{pd.Timestamp(ds.time.values[i]):%Y-%m-%d %H UTC}")
    fig.savefig(out / "check.png", dpi=100); plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None, source=None, z_era5=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cerra", default="data/cerra")
    p.add_argument("--out", default="data/pairs")
    p.add_argument("--years", nargs=2, type=int, default=[2014, 2020])
    p.add_argument("--train-years", nargs=2, type=int, default=None,
                   help="default: all but the last year (2014-2019 for 2014-2020)")
    p.add_argument("--batch", type=int, default=28, help="ERA5 time steps per fetch (memory vs. round trips)")
    p.add_argument("--consolidate", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cerra_dir, out = Path(a.cerra), Path(a.out)
    years = tuple(a.years)
    train_years = tuple(a.train_years) if a.train_years else (years[0], max(years[0], years[1] - 1))

    if a.consolidate:
        s = consolidate(out, years, train_years)
        print(json.dumps(s, indent=2)); return 0

    st = xr.open_zarr(cerra_dir / "cerra_france_static.zarr")
    if source is None:
        from weather_pipeline.datasource import build_data_source
        source = build_data_source("era5")
    if z_era5 is None:
        from weather_pipeline.statics import load_statics
        z_era5 = load_statics(source)["z"]
    e2c = Era5OnCerra(source, st.lat.values, st.lon.values, batch=a.batch)
    do_static(cerra_dir, out, e2c, z_era5)

    tags = [f"{y}{m:02d}" for y in range(years[0], years[1] + 1) for m in range(1, 13)]
    pending, reports = [], []
    for tag in tags:
        if not (cerra_dir / "months" / f"t2m_{tag}.zarr").exists():
            pending.append(tag); continue
        rep = do_month(cerra_dir, out, tag, e2c, st.lat.values, st.lon.values)
        if rep is None:
            continue
        reports.append(rep)
        logger.info("%s: %d steps, ERA5 vs CERRA corr %.3f, mean diff %+.2f K, rmse %.2f K (%ss) %s",
                    tag, rep["steps"], rep["corr"], rep["mean_diff_K"], rep["rmse_K"], rep["fetch_seconds"],
                    "PROBLEMS: " + "; ".join(rep["problems"]) if rep["problems"] else "ok")
    with open(out / "month_reports.jsonl", "a") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")
    done = len(list((out / "months").glob("pairs_*.zarr")))
    bad = [r["tag"] for r in reports if r["problems"]]
    print(f"\n{done}/{len(tags)} months built; {len(pending)} waiting for CERRA; {len(bad)} with problems {bad if bad else ''}")
    print("all months built: run again with --consolidate" if done == len(tags) else "rerun later to pick up new CERRA months")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())