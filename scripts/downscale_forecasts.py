"""
Step 6: AIFS-ENS forecasts -> the CERRA grid -> downscaled with model.pt.

    uv run --with-requirements requirements/downscaler.txt \
        python scripts/downscale_forecasts.py \
        --forecasts outputs/france_aug2025/aifs-ens1.0_*.zarr \
        --model runs/crps/model.pt --out outputs/downscaled_aug2025 --samples 4

For each forecast zarr (one initialisation) writes <out>/<name>.zarr with
    t2m_interp  (time, lead_time, ensemble, y, x)          AIFS bilinear onto the CERRA grid
    t2m_ds      (time, lead_time, ensemble, sample, y, x)  the downscaled ensemble:
                for every AIFS member, `samples` plausible fine-scale maps
and a map figure for the warmest lead time of the last initialisation.
Both fields are in K on the same curvilinear grid (2-D lat/lon coords).

Memory per init: 40 leads x 10 members x 4 samples x ~54k points x 4 B ~ 350 MB.
Finished inits are skipped on rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline.downscaler import Downscaler
from weather_pipeline.station_interp import GridInterpolator

logger = logging.getLogger("downscale")


def to_fine_grid(fc: xr.DataArray, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """fc (time, lead_time, ensemble, lat, lon) on the regular AIFS grid ->
    (time, lead_time, ensemble, y, x) on the CERRA grid, bilinear."""
    interp = GridInterpolator.build(fc["lat"].values, fc["lon"].values, lat2d.ravel(), lon2d.ravel())
    frac_out = 1 - interp.inside.mean()
    if frac_out > 0.02:
        raise ValueError(f"{frac_out:.1%} of the CERRA grid lies outside the AIFS crop: forecast region too small")
    if frac_out > 0:
        # the CERRA grid is rotated, so its corners can poke a little past the
        # AIFS box; those few points take the nearest edge value
        import dataclasses
        logger.warning("%.2f%% of CERRA points (grid corners) lie outside the AIFS crop: using edge values",
                       100 * frac_out)
        interp = dataclasses.replace(interp, inside=np.ones_like(interp.inside))
    out = interp.apply(fc)                                        # (..., station)
    return out.values.reshape(*out.shape[:-1], *lat2d.shape)


def downscale_init(fc_path: Path, model: Downscaler, out_dir: Path, samples: int, batch: int) -> Path:
    target = out_dir / Path(fc_path).name
    if target.exists():
        logger.info("skip %s (done)", target.name); return target
    ds = xr.open_zarr(fc_path)
    fc = ds["t2m"]
    if "ensemble" not in fc.dims:
        fc = fc.expand_dims(ensemble=[0])
    fc = fc.transpose("time", "lead_time", "ensemble", "lat", "lon").load()
    t0 = time.time()
    coarse = to_fine_grid(fc, model.lat, model.lon).astype("f4")   # (1, L, E, y, x)
    init = pd.Timestamp(fc["time"].values[0])
    valid = (fc["time"].values[:, None] + fc["lead_time"].values[None, :])   # (1, L)
    nI, nL, nE = coarse.shape[:3]
    out = np.empty((nI, nL, nE, samples if model.net.cfg.stochastic else 1, *model.lat.shape), "f4")
    for i in range(nI):
        for e in range(nE):
            out[i, :, e] = model.predict(coarse[i, :, e], valid[i], members=samples, batch=batch)
    res = xr.Dataset(
        {"t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), coarse),
         "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), out)},
        coords={"time": fc["time"].values, "lead_time": fc["lead_time"].values, "ensemble": fc["ensemble"].values,
                "sample": np.arange(out.shape[3]), "lat": (("y", "x"), model.lat), "lon": (("y", "x"), model.lon),
                "valid_time": (("time", "lead_time"), valid)})
    for v in ("t2m_interp", "t2m_ds"):
        res[v].attrs["units"] = "K"
    res.attrs = {"forecast": str(fc_path), "model": str(model.net.cfg), "init_time": str(init),
                 "note": "t2m_ds: for each AIFS member, `sample` plausible fine-scale maps from the downscaler"}
    res = res.chunk({"lead_time": 10, "ensemble": 1, "sample": -1, "y": -1, "x": -1})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    res.to_zarr(tmp, mode="w"); tmp.rename(target)
    d = out.mean(3) - coarse
    logger.info("%s: %d leads x %d members x %d samples in %.0fs; mean correction %+.2f K, "
                "sample spread %.2f K, downscaled range %.1f..%.1f K", target.name, nL, nE, out.shape[3],
                time.time() - t0, d.mean(), out.std(3, ddof=1).mean() if out.shape[3] > 1 else 0, out.min(), out.max())
    return target


def map_figure(path: Path, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = xr.open_zarr(path)
    hot = ds["t2m_interp"].isel(time=0).mean(("ensemble", "y", "x"))
    hours = pd.to_datetime(ds["valid_time"].isel(time=0).values).hour
    cand = np.flatnonzero(hours == 12) if (hours == 12).any() else np.arange(len(hours))
    j = int(cand[np.argmax(hot.values[cand])])
    interp = ds["t2m_interp"].isel(time=0, lead_time=j, ensemble=0).values - 273.15
    dsm = ds["t2m_ds"].isel(time=0, lead_time=j, ensemble=0).values - 273.15        # (sample, y, x)
    lo, hi = np.nanpercentile(dsm, [1, 99])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    for a, (t, f) in zip(ax, [("AIFS member 0, interpolated", interp), ("downscaled, sample 0", dsm[0]),
                                 ("downscaled, mean of samples", dsm.mean(0))]):
        m = a.pcolormesh(ds.lon, ds.lat, f, cmap="RdYlBu_r", vmin=lo, vmax=hi, shading="auto"); a.set_title(t)
        fig.colorbar(m, ax=a, label="°C")
    sd = dsm.std(0, ddof=1) if dsm.shape[0] > 1 else np.zeros_like(dsm[0])
    m = ax[3].pcolormesh(ds.lon, ds.lat, sd, cmap="magma", shading="auto"); ax[3].set_title("spread of samples")
    fig.colorbar(m, ax=ax[3], label="K")
    for a in ax:
        a.set_aspect(1 / np.cos(np.deg2rad(46)))
    vt = pd.Timestamp(ds["valid_time"].isel(time=0, lead_time=j).values)
    fig.suptitle(f"init {pd.Timestamp(ds.time.values[0]):%Y-%m-%d %H} UTC, valid {vt:%Y-%m-%d %H} UTC (+{int(ds.lead_time.values[j] / np.timedelta64(1, 'h'))} h)")
    fig.savefig(png, dpi=95); plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecasts", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=4, help="downscaled maps per AIFS member")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    model = Downscaler.load(a.model, a.device)
    logger.info("model: %s on %s, fine grid %dx%d", model.net.cfg, model.device, *model.lat.shape)
    out = Path(a.out); done = []
    for f in sorted(a.forecasts):
        done.append(downscale_init(Path(f), model, out, a.samples, a.batch))
    map_figure(done[-1], out / "map_check.png")
    print(f"\n{len(done)} initialisation(s) downscaled -> {out}/  (map: {out / 'map_check.png'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())