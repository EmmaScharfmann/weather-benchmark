"""
CLI entry point tying data -> backend -> forecast zarr together for a single
model per run. Scoring and plotting are separate, on-demand steps (see
scoring.py / plotting.py) -- run them for whichever variable you care about,
after the forecast exists.

This module only ever imports whatever backend the caller asked for, so it's
the thing each per-model uv environment actually runs. It does NOT know
about uv itself -- that's scripts/run_benchmark.py's job. Called directly:

    uv run --with-requirements requirements/aurora.txt \
        -m weather_pipeline.pipeline \
        --model aurora --init-time 2026-03-13T00:00:00 --n-steps 8

or, for the whole benchmark across models at once, see
scripts/run_benchmark.py instead.
"""

from __future__ import annotations

import argparse
import logging

from .backends import get_backend
from .datasource import build_data_source

logger = logging.getLogger(__name__)


def run_pipeline(model: str, init_time: str, n_steps: int, out_zarr: str,
                 n_members: int | None = None, source: str = "era5") -> str:
    data_source = build_data_source(source)
    backend = get_backend(model, n_members=n_members)
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
    parser.add_argument(
        "--model", required=True,
        choices=["aurora", "aurora1.5", "aurora1.5-ens", "atlas", "atlas-crps",
         "aifs1.0", "aifs1.1", "aifs-ens1.0"],
    )
    parser.add_argument("--init-time", required=True, help="e.g. 2026-03-13T00:00:00")
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument(
        "--source", default="era5", choices=["era5", "ifs"],
        help="Initial-condition source. era5: historical, reproducible, lags "
             "~a quarter. ifs: rolling ~2-week window of recent analyses, and "
             "the analysis AIFS/Aurora1.5 were fine-tuned on.",
    )
    parser.add_argument(
        "--n-members", type=int, default=None,
        help="Ensemble members (anemoi ensemble checkpoints only). Each "
             "member is a fresh stochastic draw from the same initial state.",
    )
    parser.add_argument(
        "--out", default=None, help="Output zarr path (default: outputs/<model>_forecast.zarr)"
    )
    args = parser.parse_args()

    out_zarr = args.out or f"outputs/{args.model}_forecast.zarr"
    logger.info("Starting pipeline: model=%s init_time=%s n_steps=%d", args.model, args.init_time, args.n_steps)
    run_pipeline(args.model, args.init_time, args.n_steps, out_zarr,
                 args.n_members, args.source)
    logger.info("Done. Forecast written to %s", out_zarr)


if __name__ == "__main__":
    main()