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
import logging
import os
import subprocess
from pathlib import Path

from weather_pipeline.datasource import ArraylakeERA5Source, open_era5_repo
from weather_pipeline.plotting import plot_rmse
from weather_pipeline.postprocess import load_forecast
from weather_pipeline.scoring import compute_rmse, fetch_truth

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


def run_model_isolated(model: str, init_time: str, n_steps: int, out_zarr: str) -> None:
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
    logger.info("Running '%s' in its own uv env: %s", model, " ".join(cmd))
    env = dict(os.environ)
    if model == "atlas":
        # torch-harmonics only compiles its CUDA kernels when this is set;
        # earth2studio's own install docs use it for the same package.
        # Without it the source build produces a CPU-only extension, which
        # imports fine but is much slower.
        env["FORCE_CUDA_EXTENSION"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def score_and_plot(
    model: str, out_zarr: str, variable: str, data_source: ArraylakeERA5Source
) -> None:
    """Runs in THIS process -- the shared base env. Needs xarray/pandas/
    matplotlib only, so it works regardless of which model just ran."""
    forecast = load_forecast(out_zarr, [variable])
    truth = fetch_truth(data_source, variable, forecast["valid_time"].values)
    rmse = compute_rmse(forecast, truth, variable)
    plot_rmse(rmse, variable, model, out_path=f"outputs/{model}_{variable}_rmse.png")
    logger.info("%s RMSE (%s):\n%s", model, variable, rmse)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Run + score multiple Earth2Studio models, each isolated in its own uv env"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=["aurora", "atlas", "aifs1.0", "aifs1.1", "aifs-ens1.0"],
    )
    parser.add_argument("--init-time", required=True, help="e.g. 2026-03-13T00:00:00")
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--variable", default="t2m")
    args = parser.parse_args()

    single, pl = open_era5_repo()
    data_source = ArraylakeERA5Source(single, pl)

    failed = []
    for model in args.models:
        out_zarr = f"outputs/{model}_forecast.zarr"
        try:
            run_model_isolated(model, args.init_time, args.n_steps, out_zarr)
        except subprocess.CalledProcessError:
            logger.exception("Model '%s' failed inside its own env -- skipping its scoring", model)
            failed.append(model)
            continue
        score_and_plot(model, out_zarr, args.variable, data_source)

    if failed:
        logger.warning("Finished with failures: %s", failed)
    else:
        logger.info("Benchmark complete for all models: %s", args.models)


if __name__ == "__main__":
    main()