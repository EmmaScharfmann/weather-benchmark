"""
Measure the regridding noise floor.

Every AIFS forecast in this pipeline makes a round trip: ERA5 at 0.25 deg
-> the model's N320 reduced Gaussian grid -> back to 0.25 deg for scoring.
Aurora and Atlas never leave 0.25 deg, so any error that round trip
introduces is charged to AIFS alone.

This script isolates that error with no model involved: take an ERA5
analysis field, push it through exactly the same to_n320 / to_latlon
functions the backend uses, and difference it against the original. What
comes back is a pure interpolation artifact -- an error floor below which
no AIFS score can go, and a map of WHERE that floor is high.

Expect it to be worst at sharp gradients (coastlines, the sea-ice edge,
steep orography) and toward the poles, where N320 rows have fewest points.
If the resulting map resembles the polar/coastal banding in the models'
bias maps, that banding is substantially this, not model error.

Output is written in the same form as the per-model bias fields, so it
appears as an extra panel in the report alongside the models -- a control
you can read against them directly.

Usage:
    uv run --with-requirements requirements/aifs1.0.txt \\
        scripts/regrid_diagnostic.py --time 2026-03-13T00:00:00 --variable t2m
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", required=True, help="Any analysis time, e.g. 2026-03-13T00:00:00")
    parser.add_argument("--variable", default="t2m")
    args = parser.parse_args()

    # imported here so the module docstring is readable without the heavy
    # anemoi/earthkit stack installed
    from weather_pipeline.anemoi_backend import to_latlon, to_n320
    from weather_pipeline.datasource import ArraylakeERA5Source, open_era5_repo

    single, pl = open_era5_repo()
    source = ArraylakeERA5Source(single, pl)

    t = pd.Timestamp(args.time)
    logger.info("Fetching ERA5 %s at %s", args.variable, t)
    da = source([t], [args.variable]).sel(variable=args.variable, drop=True).isel(time=0)

    original = np.asarray(da.values, dtype="f8")
    logger.info("Round-tripping 0.25deg -> N320 -> 0.25deg")
    roundtrip = to_latlon(to_n320(original))
    diff = roundtrip - original

    lat = np.asarray(da["lat"].values, dtype="f8")
    w = np.cos(np.deg2rad(lat))[:, None]
    rmse = float(np.sqrt((w * diff**2).sum() / (w * np.ones_like(diff)).sum()))

    polar = np.abs(lat) > 60
    rmse_polar = float(np.sqrt(
        (w[polar] * diff[polar] ** 2).sum() / (w[polar] * np.ones_like(diff[polar])).sum()
    ))
    rmse_tropics = float(np.sqrt(
        (w[~polar] * diff[~polar] ** 2).sum() / (w[~polar] * np.ones_like(diff[~polar])).sum()
    ))

    print(f"\nRegridding noise floor for {args.variable} ({args.time}):")
    print(f"  global      RMSE = {rmse:.4f}")
    print(f"  |lat| > 60  RMSE = {rmse_polar:.4f}")
    print(f"  |lat| <= 60 RMSE = {rmse_tropics:.4f}")
    print(f"  max |diff|       = {np.abs(diff).max():.4f}")
    print(
        "\nCompare the global figure against the AIFS models' RMSE: if it is a "
        "large fraction of it, the comparison with Aurora/Atlas is measuring "
        "interpolation as much as forecast skill.\n"
    )

    import xarray as xr
    out = xr.DataArray(
        diff.astype("f4"),
        coords={"latitude": da["lat"].values, "longitude": da["lon"].values},
        dims=("latitude", "longitude"),
        name=f"{args.variable}_mean_bias",
    )
    fields = REPO_ROOT / "outputs/fields"
    fields.mkdir(parents=True, exist_ok=True)
    slug = t.strftime("%Y%m%dT%H%M")
    path = fields / f"REGRID-CONTROL_{args.variable}_{slug}_bias.zarr"
    out.to_dataset().to_zarr(path, mode="w")
    logger.info("Wrote %s -- it will appear as a control panel in the report", path)


if __name__ == "__main__":
    main()