"""
Benchmark orchestrator. Runs one or more models, each in its OWN
uv-managed environment (so aurora's torch pin never has to coexist
in-process with aifs's), then scores every model's forecast against
ERA5 truth using the shared base env this script itself runs in.

This script never imports torch, earth2studio, or any model backend
directly -- it only shells out to `uv run` per model and then reads back
the zarr file each run produces. That's what keeps it runnable in the
lightweight base env (numpy/xarray/matplotlib only) regardless of which
models are in the --models list.

Usage:
    uv run scripts/run_benchmark.py \
        --models aurora aifs1.0 atlas \
        --init-time 2026-03-13T00:00:00 --n-steps 8 --variable t2m

Add a model later by adding requirements/<name>.txt and a
BACKEND_FACTORIES entry in weather_pipeline/backends.py -- nothing in this
script needs to change.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from weather_pipeline.datasource import ArraylakeERA5Source, open_era5_repo
from weather_pipeline.postprocess import load_forecast
from weather_pipeline.scoring import (
    describe_metrics, fetch_truth, mean_bias_field,
)
from weather_pipeline.scoring import score_case as _score_case

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

# Wheel indexes for the CUDA-specific builds the model envs need:
#   - torch's CUDA build (cu126) isn't on PyPI, only on pytorch's own index
#   - natten publishes prebuilt wheels per torch+CUDA combo on its own host;
#     without this it falls back to compiling from source, which appears to
#     hang for a very long time and needs nvcc/CUDA headers to even succeed
#
# Passed as `uv run` CLI FLAGS rather than set in pyproject.toml's [tool.uv]
# table, because config-file settings proved unreliable for the nested
# `--with-requirements` resolutions below -- honored on some runs, silently
# ignored on others, producing a confusing "no version of torch==2.13.0+cu126"
# even though that wheel demonstrably exists on the index. CLI flags are what
# uv's own error message points at, and they take precedence over config.
# Hardcoded here so nobody running the benchmark has to know or type them.
#
# If this repo is ever used on a machine with a different CUDA version, these
# URLs and the matching torch/natten pins in requirements/*.txt need to change
# together.
UV_INDEX_ARGS = [
    "--extra-index-url", "https://download.pytorch.org/whl/cu126",
    "--find-links", "https://whl.natten.org",
    # pytorch's index also mirrors common packages (certifi,
    # importlib-metadata, ...) at stale versions. uv's default is to take each
    # package from the first index that has it, which let those stale copies
    # win and made nvidia-physicsnemo's real requirements look unsatisfiable.
    # This makes uv compare versions across all indexes per package instead.
    "--index-strategy", "unsafe-best-match",
]

# Per-model extra `uv run` flags, for packages that ship a precompiled
# extension built against a different torch than the one we pin.
#
# torch-harmonics (atlas): its wheel bundles a compiled `_C` extension. If
# that was built against a different torch ABI than ours, importing it dies
# with "undefined symbol: _ZNR5torch7Library4_def...". Forcing a source
# build makes it compile against the torch actually installed in this env;
# --no-build-isolation-package lets the build see that torch (it needs
# torch's headers at build time, which an isolated build env wouldn't have).
#
# Expect this build to be slow. If it fails for lack of nvcc/CUDA headers,
# the fallback is a persistent venv for atlas rather than the ephemeral
# --with-requirements path.
# AIFS no longer needs any build flags: requirements/aifs*.txt pins the
# official prebuilt flash-attn wheel by URL, so nothing compiles. (Kept the
# lookup prefix-aware anyway -- if a future model does need build flags,
# this is where they go.)
EXTRA_UV_ARGS: dict[str, list[str]] = {
    "atlas": [
        "--no-binary-package", "torch-harmonics",
        "--no-build-isolation-package", "torch-harmonics",
    ],
}


def _extra_uv_args(model: str) -> list[str]:
    """Exact model name first, then prefix -- so every aifs* variant picks
    up the shared flash-attn handling without repeating it per version."""
    if model in EXTRA_UV_ARGS:
        return EXTRA_UV_ARGS[model]
    for prefix, args in EXTRA_UV_ARGS.items():
        if model.startswith(prefix):
            return args
    return []


def init_slug(init_time: str) -> str:
    """Filesystem-safe form of an init time, e.g. 2026-03-13T00:00:00 ->
    20260313T0000. Forecast paths include it so multiple init times coexist
    instead of overwriting each other."""
    return pd.Timestamp(init_time).strftime("%Y%m%dT%H%M")


def forecast_path(model: str, init_time: str) -> str:
    return f"outputs/forecasts/{model}_{init_slug(init_time)}.zarr"


def _manifest_path(out_zarr: str) -> Path:
    """Sidecar describing what produced a forecast zarr.

    A sidecar rather than zarr attrs because the two write paths differ:
    Earth2Studio's ZarrBackend owns the store for aurora/atlas, while the
    anemoi backend writes its own. A file next to the store works for both
    without either backend needing to cooperate."""
    return REPO_ROOT / (out_zarr + ".manifest.json")


def existing_forecast_matches(model: str, init_time: str, n_steps: int, out_zarr: str,
                              n_members: int | None = None) -> bool:
    """True if a usable forecast for exactly this request is already on disk.

    Checks the manifest rather than mere existence of the zarr: a store left
    behind by a different init time, a shorter run, or a crashed run
    mid-write would otherwise be silently reused and scored as if it were
    the requested forecast."""
    store = REPO_ROOT / out_zarr
    manifest = _manifest_path(out_zarr)
    if not store.exists() or not manifest.exists():
        return False
    try:
        meta = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Unreadable manifest at %s; re-running", manifest)
        return False

    want = {"model": model, "init_time": init_time, "n_steps": n_steps,
            "n_members": n_members}
    got = {k: meta.get(k) for k in want}
    if got != want:
        logger.info("Existing forecast at %s is for %s, want %s; re-running",
                    out_zarr, got, want)
        return False
    return True


def write_manifest(model: str, init_time: str, n_steps: int, out_zarr: str,
                   n_members: int | None = None) -> None:
    """Written only AFTER the run succeeds, so a crashed run leaves no
    manifest and is never mistaken for a complete forecast."""
    _manifest_path(out_zarr).write_text(json.dumps({
        "model": model,
        "init_time": init_time,
        "n_steps": n_steps,
        "n_members": n_members,
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2))


def run_model_isolated(model: str, init_time: str, n_steps: int, out_zarr: str,
                       n_members: int | None = None) -> None:
    """Runs one model's forecast to completion in its own uv environment.

    Each call is a fresh `uv run` invocation: uv resolves & (on first run)
    installs the base project deps plus this model's requirements file
    into an isolated environment, then executes the pipeline inside it.
    Later runs reuse uv's cache, so this is only slow the first time per
    model.

    The CUDA wheel indexes (UV_INDEX_ARGS above) are baked into the command
    so callers never have to set env vars or pass flags themselves."""
    req_file = REQUIREMENTS_DIR / f"{model}.txt"
    if not req_file.exists():
        raise FileNotFoundError(
            f"No requirements file for model '{model}': {req_file}. "
            "Add one alongside the others in requirements/ first."
        )

    cmd = [
        "uv", "run",
        *UV_INDEX_ARGS,
        *_extra_uv_args(model),
        "--with-requirements", str(req_file),
        "-m", "weather_pipeline.pipeline",
        "--model", model,
        "--init-time", init_time,
        "--n-steps", str(n_steps),
        "--out", out_zarr,
    ]
    if n_members is not None:
        cmd += ["--n-members", str(n_members)]
    logger.info("Running '%s' in its own uv env: %s", model, " ".join(cmd))
    env = dict(os.environ)
    if model == "atlas":
        # torch-harmonics only compiles its CUDA kernels when this is set;
        # earth2studio's own install docs use it for the same package.
        # Without it the source build produces a CPU-only extension, which
        # imports fine but is much slower.
        env["FORCE_CUDA_EXTENSION"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    write_manifest(model, init_time, n_steps, out_zarr, n_members)


def score_case(
    model: str, init_time: str, out_zarr: str, variable: str,
    data_source: ArraylakeERA5Source, metrics="all",
) -> None:
    """Score one (model, init_time) and write the pieces the report needs.

    Runs in THIS process -- the shared base env -- so it needs only
    xarray/pandas, regardless of which model produced the forecast.

    Writes THREE things per case:
      * accumulators CSV -- the linearly-averageable inner values. This is
        what the report averages across init times before finalising, which
        is the only correct way to aggregate RMSE-like metrics.
      * metrics CSV -- the finalised values for this single case, for when
        you just want to look at one forecast.
      * bias field -- a small 2-D time-mean error map (zarr), cached so the
        report can draw spatial panels without re-reading forecasts or
        re-fetching truth from Arraylake."""
    forecast = load_forecast(out_zarr, [variable])
    truth = fetch_truth(data_source, variable, forecast["valid_time"].values)
    acc, fin = _score_case(forecast, truth, variable, metrics)

    slug = init_slug(init_time)
    scores_dir = REPO_ROOT / "outputs/scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    acc.to_csv(scores_dir / f"{model}_{variable}_{slug}_accumulators.csv")
    fin.to_csv(scores_dir / f"{model}_{variable}_{slug}_metrics.csv")

    fields_dir = REPO_ROOT / "outputs/fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    bias = mean_bias_field(forecast, truth, variable)
    # zarr, not netCDF: zarr is already a base dependency (the forecasts are
    # zarr), whereas writing netCDF would need netCDF4/h5netcdf installed
    # just for these few small 2-D fields.
    bias.to_dataset().to_zarr(
        fields_dir / f"{model}_{variable}_{slug}_bias.zarr", mode="w"
    )

    logger.info("%s @ %s metrics (%s):\n%s", model, init_time, variable, fin)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Run + score Earth2Studio/anemoi models, each isolated in its own uv env"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=["aurora", "aurora1.5", "aurora1.5-ens", "atlas", "aifs1.0", "aifs1.1", "aifs-ens1.0"],
    )
    parser.add_argument(
        "--init-times", nargs="+", required=True,
        help="One or more init times, e.g. 2026-03-13T00:00:00 2026-03-14T00:00:00. "
             "Every model is run for every init time; scores are aggregated "
             "across them by scripts/make_report.py.",
    )
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument(
        "--n-members", type=int, default=None,
        help="Ensemble members for checkpoints that sample stochastically "
             "(aifs-ens*). Each member is a fresh draw from the same initial "
             "state, so this multiplies runtime by n_members. Deterministic "
             "models reject values > 1 rather than returning identical copies.",
    )
    parser.add_argument("--variable", default="t2m")
    parser.add_argument(
        "--metrics", default="all",
        help="Comma-separated metric names, or 'all' (default). Ensemble-only "
             "metrics are skipped automatically for deterministic forecasts "
             "under 'all', but named explicitly they raise. See --list-metrics.",
    )
    parser.add_argument("--list-metrics", action="store_true",
                        help="Print available metrics and exit.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run models even when a matching forecast already exists.",
    )
    args = parser.parse_args()

    if args.list_metrics:
        print("Available metrics:\n" + describe_metrics())
        return

    single, pl = open_era5_repo()
    data_source = ArraylakeERA5Source(single, pl)

    failed = []
    for init_time in args.init_times:
        for model in args.models:
            out_zarr = forecast_path(model, init_time)
            label = f"{model} @ {init_time}"
            try:
                # Models whose checkpoints sample stochastically, so
                # repeated runs give independent members.
                stochastic = model.startswith("aifs-ens") or model.endswith("-ens")
                members = args.n_members if stochastic else None
                if not args.force and existing_forecast_matches(
                    model, init_time, args.n_steps, out_zarr, members
                ):
                    logger.info("Reusing existing forecast for %s (use --force to re-run)", label)
                else:
                    run_model_isolated(model, init_time, args.n_steps, out_zarr, members)
            except subprocess.CalledProcessError:
                logger.exception("%s failed inside its own env -- skipping its scoring", label)
                failed.append(label)
                continue

            try:
                score_case(model, init_time, out_zarr, args.variable,
                           data_source, args.metrics)
            except Exception:
                logger.exception("%s produced a forecast but scoring failed", label)
                failed.append(f"{label} (scoring)")

    if failed:
        logger.warning("Finished with failures: %s", failed)
    else:
        logger.info(
            "Complete: %d model(s) x %d init time(s). Build the comparison with:\n"
            "    uv run scripts/make_report.py --variable %s",
            len(args.models), len(args.init_times), args.variable,
        )


if __name__ == "__main__":
    main()