"""
CLI entry point tying data -> backend -> forecast zarr together for a single
model per run. Scoring and plotting are separate, on-demand steps (see
scoring.py / plotting.py, or example_usage.py for the full flow) -- run them
for whichever variable you care about, after the forecast exists.

Usage:
    python -m earth2studio_pipeline.pipeline \
        --model aurora --init-time 2026-03-13T00:00:00 --n-steps 8
"""

from __future__ import annotations

import argparse
import logging

from .backends import get_backend
from .datasource import ArraylakeERA5Source, open_era5_repo

logger = logging.getLogger(__name__)


def run_pipeline(model: str, init_time: str, n_steps: int, out_zarr: str) -> str:
    single, pl = open_era5_repo()
    data_source = ArraylakeERA5Source(single, pl)
    backend = get_backend(model)
    backend.run(data_source, init_time, n_steps, out_zarr)
    return out_zarr


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Run one Earth2Studio model on Arraylake ERA5 initial conditions"
    )
    parser.add_argument("--model", required=True, choices=["aurora", "aifs1.0", "aifs1.1", "aifs2", "atlas"])
    parser.add_argument("--init-time", required=True, help="e.g. 2026-03-13T00:00:00")
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument(
        "--out", default=None, help="Output zarr path (default: outputs/<model>_forecast.zarr)"
    )
    args = parser.parse_args()

    out_zarr = args.out or f"outputs/{args.model}_forecast.zarr"
    logger.info("Starting pipeline: model=%s init_time=%s n_steps=%d", args.model, args.init_time, args.n_steps)
    run_pipeline(args.model, args.init_time, args.n_steps, out_zarr)
    logger.info("Done. Forecast written to %s", out_zarr)


if __name__ == "__main__":
    main()