"""Apply a trained downscaler to coarse ensemble forecasts.

Usage::

    python scripts/downscale_forecasts.py --config config.yaml --model runs/crps/model.pt
    
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import load_config
from downscale.constants import MAX_FRACTION_OUTSIDE
from downscale.downscaler import Downscaler
from downscale.station_interp import GridInterpolator

logger = logging.getLogger("downscale")


def to_fine_grid(forecast: xr.DataArray, lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Interpolate a forecast onto the fine grid.

    A rotated fine grid can have corners slightly outside the forecast domain;
    a small fraction is allowed and takes edge values.

    :param forecast: Forecast over ``(time, lead_time, ensemble, lat, lon)``.
    :type forecast: xarray.DataArray
    :param lat2d: Fine grid latitudes with shape ``(y, x)``.
    :type lat2d: numpy.ndarray
    :param lon2d: Fine grid longitudes with shape ``(y, x)``.
    :type lon2d: numpy.ndarray
    :returns: Values over ``(time, lead_time, ensemble, y, x)``.
    :rtype: numpy.ndarray
    :raises ValueError: If too much of the fine grid lies outside the forecast domain.
    """
    interp = GridInterpolator.build(
        forecast["lat"].values, forecast["lon"].values, lat2d.ravel(), lon2d.ravel()
    )
    fraction_outside = 1 - interp.inside.mean()
    if fraction_outside > MAX_FRACTION_OUTSIDE:
        raise ValueError(
            f"{fraction_outside:.1%} of the fine grid lies outside the forecast domain: "
            "the forecast region is too small"
        )
    if fraction_outside > 0:
        logger.warning(
            "%.2f%% of fine grid points lie outside the forecast domain and take edge values",
            100 * fraction_outside,
        )
        interp = dataclasses.replace(interp, inside=np.ones_like(interp.inside))
    out = interp.apply(forecast)
    return out.values.reshape(*out.shape[:-1], *lat2d.shape)


def downscale_init(
    forecast_path: Path, model: Downscaler, out_dir: Path, variable: str, samples: int, batch: int
) -> Path:
    """Downscale one initialisation.

    :param forecast_path: Forecast store for a single initialisation.
    :type forecast_path: pathlib.Path
    :param model: Trained downscaler.
    :type model: weather_pipeline.downscaler.Downscaler
    :param out_dir: Output directory.
    :type out_dir: pathlib.Path
    :param variable: Variable to downscale.
    :type variable: str
    :param samples: Fine-scale maps per coarse member.
    :type samples: int
    :param batch: Lead times processed at once.
    :type batch: int
    :returns: Path to the downscaled store.
    :rtype: pathlib.Path
    :raises RuntimeError: If the forecast has no completion marker.
    """
    target = out_dir / Path(forecast_path).name
    if not Path(str(forecast_path).rstrip("/") + ".done").exists():
        raise RuntimeError(
            f"{Path(forecast_path).name} has no .done marker: the forecast run is unfinished, and a "
            "partial store can hold fewer members than requested"
        )
    ds = xr.open_zarr(forecast_path)
    n_members = ds.sizes.get("ensemble", 1)
    if target.exists():
        existing = xr.open_zarr(target).sizes["ensemble"]
        if existing == n_members:
            logger.info("skipping %s, already downscaled with %d members", target.name, existing)
            return target
        logger.warning(
            "%s holds %d members but the forecast has %d: redoing it", target.name, existing, n_members
        )
        shutil.rmtree(target)

    forecast = ds[variable]
    if "ensemble" not in forecast.dims:
        forecast = forecast.expand_dims(ensemble=[0])
    forecast = forecast.transpose("time", "lead_time", "ensemble", "lat", "lon").load()
    started = time.time()
    coarse = to_fine_grid(forecast, model.lat, model.lon).astype("f4")
    valid = forecast["time"].values[:, None] + forecast["lead_time"].values[None, :]
    n_init, n_lead, n_ensemble = coarse.shape[:3]
    n_samples = samples if model.net.cfg.stochastic else 1
    fine = np.empty((n_init, n_lead, n_ensemble, n_samples, *model.lat.shape), "f4")
    for i in range(n_init):
        for member in range(n_ensemble):
            fine[i, :, member] = model.predict(coarse[i, :, member], valid[i], members=samples, batch=batch)

    result = xr.Dataset(
        {
            "t2m_interp": (("time", "lead_time", "ensemble", "y", "x"), coarse),
            "t2m_ds": (("time", "lead_time", "ensemble", "sample", "y", "x"), fine),
        },
        coords={
            "time": forecast["time"].values,
            "lead_time": forecast["lead_time"].values,
            "ensemble": forecast["ensemble"].values,
            "sample": np.arange(n_samples),
            "lat": (("y", "x"), model.lat),
            "lon": (("y", "x"), model.lon),
            "valid_time": (("time", "lead_time"), valid),
        },
    )
    for name in ("t2m_interp", "t2m_ds"):
        result[name].attrs["units"] = "K"
    result.attrs = {
        "forecast": str(forecast_path),
        "model": str(model.net.cfg),
        "init_time": str(pd.Timestamp(forecast["time"].values[0])),
        "note": "t2m_ds holds, for each coarse member, several plausible fine-scale maps",
    }
    result = result.chunk({"lead_time": 10, "ensemble": 1, "sample": -1, "y": -1, "x": -1})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    result.to_zarr(tmp, mode="w")
    tmp.rename(target)
    correction = fine.mean(3) - coarse
    logger.info(
        "%s: %d lead times x %d members x %d samples in %.0fs; mean correction %+.2f K, "
        "sample spread %.2f K, range %.1f..%.1f K",
        target.name, n_lead, n_ensemble, n_samples, time.time() - started, correction.mean(),
        fine.std(3, ddof=1).mean() if n_samples > 1 else 0.0, fine.min(), fine.max(),
    )
    return target


def map_figure(path: Path, png: Path) -> None:
    """Draw the coarse and downscaled fields for the warmest lead time.

    :param path: Downscaled store.
    :type path: pathlib.Path
    :param png: Destination image.
    :type png: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xr.open_zarr(path)
    mean_field = ds["t2m_interp"].isel(time=0).mean(("ensemble", "y", "x"))
    hours = pd.to_datetime(ds["valid_time"].isel(time=0).values).hour
    candidates = np.flatnonzero(hours == 12) if (hours == 12).any() else np.arange(len(hours))
    index = int(candidates[np.argmax(mean_field.values[candidates])])
    coarse = ds["t2m_interp"].isel(time=0, lead_time=index, ensemble=0).values - 273.15
    fine = ds["t2m_ds"].isel(time=0, lead_time=index, ensemble=0).values - 273.15
    low, high = np.nanpercentile(fine, [1, 99])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    panels = [
        ("coarse member, interpolated", coarse),
        ("downscaled, sample 1", fine[0]),
        ("downscaled, mean of samples", fine.mean(0)),
    ]
    for axis, (title, field) in zip(ax, panels):
        mesh = axis.pcolormesh(ds.lon, ds.lat, field, cmap="RdYlBu_r", vmin=low, vmax=high, shading="auto")
        axis.set_title(title)
        fig.colorbar(mesh, ax=axis, label="degC")
    spread = fine.std(0, ddof=1) if fine.shape[0] > 1 else np.zeros_like(fine[0])
    mesh = ax[3].pcolormesh(ds.lon, ds.lat, spread, cmap="magma", shading="auto")
    ax[3].set_title("spread of samples")
    fig.colorbar(mesh, ax=ax[3], label="K")
    mid_lat = float(np.mean(ds.lat.values))
    for axis in ax:
        axis.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    valid = pd.Timestamp(ds["valid_time"].isel(time=0, lead_time=index).values)
    lead_hours = int(ds.lead_time.values[index] / np.timedelta64(1, "h"))
    fig.suptitle(
        f"initialised {pd.Timestamp(ds.time.values[0]):%Y-%m-%d %H} UTC, "
        f"valid {valid:%Y-%m-%d %H} UTC (+{lead_hours} h)"
    )
    fig.savefig(png, dpi=95)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--forecasts", nargs="+", default=None, help="overrides paths.forecasts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=None, help="overrides paths.downscaled")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    forecasts = sorted(args.forecasts) if args.forecasts else sorted(
        str(p) for p in Path(cfg.paths.forecasts).glob("*.zarr")
    )
    if not forecasts:
        parser.error("no forecasts found; pass --forecasts or set paths.forecasts")
    out = Path(args.out or cfg.paths.downscaled)
    samples = args.samples or cfg.inference.samples
    batch = args.batch_size or cfg.inference.batch_size

    model = Downscaler.load(args.model, args.device)
    logger.info("model %s on %s, fine grid %dx%d", model.net.cfg, model.device, *model.lat.shape)
    done = [
        downscale_init(Path(f), model, out, cfg.coarse.variable, samples, batch) for f in forecasts
    ]
    map_figure(done[-1], out / "map_check.png")
    print(f"\n{len(done)} initialisation(s) downscaled into {out}/ (map: {out / 'map_check.png'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())