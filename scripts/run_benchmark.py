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
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline.datasource import build_data_source
from weather_pipeline import wbx_scoring as wbx

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
# Models whose checkpoints sample stochastically, so repeated runs from the
# same initial condition give independent members and --n-members is
# meaningful. Listed explicitly rather than inferred from the name: Atlas
# is generative (stochastic interpolant) despite having no "-ens" suffix,
# and a deterministic model asked for members would silently return
# identical copies with zero spread.
#
# run_benchmark cannot import backends to ask (that would pull in torch),
# so this list has to be kept in step with backends.BACKEND_FACTORIES.
STOCHASTIC_MODELS = {"aifs-ens1.0", "aurora1.5-ens", "atlas", "atlas-crps"}

# Each model's NATIVE step length in hours. Earth2Studio's Aurora 1.5
# wrapper emits hourly lead times (it queries the 6-hour autoregressive
# step at t+1h..t+6h before advancing the state, so the finer output costs
# no extra model evaluations); everything else steps 6-hourly.
MODEL_STEP_HOURS = {
    "aurora": 6, "atlas": 6, "atlas-crps": 6,
    "aifs1.0": 6, "aifs1.1": 6, "aifs-ens1.0": 6,
    "aurora1.5": 1, "aurora1.5-ens": 1,
}

# --n-steps counts steps of THIS cadence, for every model. Without the
# conversion below, --n-steps 8 means 48 h for AIFS but only 8 h for
# Aurora 1.5, and the scores end up on incomparable lead-time axes.
BENCHMARK_STEP_HOURS = 6


def native_steps(model: str, n_steps: int) -> int:
    """Convert benchmark steps (6 h each) into the model's own steps."""
    step = MODEL_STEP_HOURS.get(model, BENCHMARK_STEP_HOURS)
    steps = n_steps * BENCHMARK_STEP_HOURS // step
    if step != BENCHMARK_STEP_HOURS:
        logger.info(
            "%s steps every %d h: %d benchmark step(s) (%d h) -> %d native step(s)",
            model, step, n_steps, n_steps * BENCHMARK_STEP_HOURS, steps,
        )
    return steps

EXTRA_UV_ARGS: dict[str, list[str]] = {
    # Both Atlas variants need the torch-harmonics source build.
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
                              n_members: int | None = None, source: str = "era5") -> bool:
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
            "n_members": n_members, "source": source}
    got = {k: meta.get(k) for k in want}
    if got != want:
        logger.info("Existing forecast at %s is for %s, want %s; re-running",
                    out_zarr, got, want)
        return False
    return True


def write_manifest(model: str, init_time: str, n_steps: int, out_zarr: str,
                   n_members: int | None = None, source: str = "era5") -> None:
    """Written only AFTER the run succeeds, so a crashed run leaves no
    manifest and is never mistaken for a complete forecast."""
    _manifest_path(out_zarr).write_text(json.dumps({
        "model": model,
        "init_time": init_time,
        "n_steps": n_steps,
        "n_members": n_members,
        "source": source,
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2))


def run_model_isolated(model: str, init_time: str, n_steps: int, out_zarr: str,
                       n_members: int | None = None, source: str = "era5") -> None:
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
    cmd += ["--source", source]
    # Clear any existing store first. Earth2Studio's ZarrBackend OPENS an
    # existing zarr rather than replacing it, so a store left by a previous
    # run with a different number of steps keeps its old lead_time axis and
    # the run dies partway through with
    #   "Attempting to set a selection of 1 elements with an array of N"
    # at exactly the step where the old axis ran out.
    store = REPO_ROOT / out_zarr
    if store.exists():
        logger.info("Removing existing forecast store %s before re-running", store)
        shutil.rmtree(store, ignore_errors=True)
    manifest = _manifest_path(out_zarr)
    if manifest.exists():
        manifest.unlink()

    logger.info("Running '%s' in its own uv env: %s", model, " ".join(cmd))
    env = dict(os.environ)
    if model.startswith("atlas"):
        # Atlas asks for TF32 cuBLAS for inference performance -- it warns
        # on every load without it, and NVIDIA's own examples set it.
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    if model == "atlas":
        # torch-harmonics only compiles its CUDA kernels when this is set;
        # earth2studio's own install docs use it for the same package.
        # Without it the source build produces a CPU-only extension, which
        # imports fine but is much slower.
        env["FORCE_CUDA_EXTENSION"] = "1"
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    write_manifest(model, init_time, n_steps, out_zarr, n_members, source)


def score_case(
    model: str, init_time: str, out_zarr: str, variables: list[str],
    data_source, metrics="all", stride_hours: int = 6,
    region: str = "global", bin_regions: bool = False,
    climatology: str | None = None,
) -> None:
    """Score one (model, init_time) with WeatherBench-X.

    Runs in the BASE environment -- WB-X needs only xarray/numpy/beam, no
    torch -- so it works regardless of which model produced the forecast.

    Writes two things per case:
      * an AggregationState (zarr). NOT finalised metric values: combining
        cases means summing states and computing metrics once at the end,
        and storing finished RMSEs would make that impossible afterwards.
      * a small 2-D mean-bias field, cached so the report can draw spatial
        panels without re-reading forecasts or re-fetching truth.

    Also writes a per-case CSV of finalised values, purely for eyeballing a
    single forecast -- the report ignores it and uses the states.
    """
    # Models differ in what they predict: Aurora 1 emits four surface
    # variables, AIFS a dozen, and only AIFS and Aurora 1.5 produce
    # precipitation. Score what this model has and say what it lacks,
    # rather than failing the whole case over one missing variable.
    available = set(xr.open_zarr(out_zarr).data_vars)
    wanted = [v for v in variables if v in available]
    missing = [v for v in variables if v not in available]
    if missing:
        logger.info("%s does not predict %s -- scoring %s", model, missing, wanted)
    if not wanted:
        logger.warning("%s predicts none of %s; nothing to score", model, variables)
        return

    forecast = wbx.open_forecast(out_zarr, wanted)
    forecast = subsample_leads_wbx(forecast, stride_hours)

    # Subset BEFORE fetching truth: targets are built on the forecast's own
    # coordinates, so restricting the forecast restricts the truth fetch
    # too -- a regional run reads only the grid points it scores.
    if region != "global":
        box = wbx.parse_region(region)
        forecast = wbx.subset_region(forecast, box)
        logger.info("Region '%s' %s -> %d x %d grid points", region, box,
                    forecast.sizes["latitude"], forecast.sizes["longitude"])

    # One truth fetch for every variable -- that is the expensive part, and
    # the alternative (a fetch per variable) would re-read the same valid
    # times repeatedly.
    targets = wbx.fetch_targets(data_source, forecast, wanted)
    ensemble = wbx.has_ensemble(forecast)

    slug = init_slug(init_time)
    scores_dir = REPO_ROOT / "outputs/scores"
    fields_dir = REPO_ROOT / "outputs/fields"
    scores_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    # A state PER VARIABLE, not one covering all of them: the report globs
    # by variable, and models cover different variable sets, so keeping
    # them separate means a model missing one variable still contributes to
    # the others.
    clim = wbx.load_climatology(climatology, wanted) if climatology else None
    # Binning computes EVERY region in one pass, adding a `region`
    # dimension -- unlike --region, which throws the rest of the globe away.
    # Kept under a separate tag because the resulting states have an extra
    # dimension the global report does not expect.
    bins = wbx.build_binning(regions=True) if bin_regions else None

    for variable in wanted:
        state, results, _ = wbx.score_case(
            forecast, targets, [variable], climatology=clim, bin_by=bins
        )
        df = wbx.to_dataframe(results)

        # Region goes in the filename so global and regional scores live
        # side by side and the report never mixes them by accident.
        tag = variable if region == "global" else f"{variable}@{region}"

        # Binned states go in their OWN directory rather than carrying a
        # marker in the filename: zarr/fsspec treat "#" as a URL fragment
        # separator and silently truncate the path at it, so a "#binned"
        # suffix produced files named for everything BEFORE the marker.
        out_dir = scores_dir / "binned" if bin_regions else scores_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        state_path = out_dir / f"{model}_{tag}_{slug}_state.zarr"
        shutil.rmtree(state_path, ignore_errors=True)
        # Record whether this was an ensemble: metric_values needs the same
        # metric objects the state was built with, and the report reads the
        # flag back rather than guessing from the stored statistic names.
        wbx.save_state(state, state_path, ensemble=ensemble)
        df.to_csv(out_dir / f"{model}_{tag}_{slug}_metrics.csv")

        # Bias map from the SAME arrays the metrics used -- no second load
        # of the forecast, no second truth fetch, and no risk of the map
        # covering different lead times or a different region.
        bias = wbx.mean_bias_field(forecast, targets, variable)
        bias_path = fields_dir / f"{model}_{tag}_{slug}_bias.zarr"
        shutil.rmtree(bias_path, ignore_errors=True)
        bias.to_dataset().to_zarr(bias_path, mode="w")

        logger.info("%s @ %s (%s):\n%s", model, init_time, variable, df)


def subsample_leads_wbx(forecast, stride_hours: int):
    """Same cadence filter as subsample_leads, on WB-X's layout where lead
    time is its own dimension rather than folded into valid_time."""
    hours = (forecast["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    # h=0 is the initial condition, not a forecast: Earth2Studio's iterator
    # yields it first, and scoring it gives a meaningless perfect zero.
    keep = (hours > 0) & (hours % stride_hours == 0)
    if keep.all():
        return forecast
    logger.info("Scoring %d of %d lead times (%d h cadence)",
                int(keep.sum()), len(hours), stride_hours)
    return forecast.isel(lead_time=np.flatnonzero(keep))


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
        choices=["aurora", "aurora1.5", "aurora1.5-ens", "atlas", "atlas-crps",
         "aifs1.0", "aifs1.1", "aifs-ens1.0"],
    )
    parser.add_argument(
        "--init-times", nargs="+", required=True,
        help="One or more init times, e.g. 2026-03-13T00:00:00 2026-03-14T00:00:00. "
             "Every model is run for every init time; scores are aggregated "
             "across them by scripts/make_report.py.",
    )
    parser.add_argument(
        "--n-steps", type=int, required=True,
        help="Forecast length in 6-HOUR steps, the same for every model. "
             "Models with a different native step (Aurora 1.5 runs hourly) "
             "are converted automatically, so --n-steps 8 is +48 h "
             "everywhere and the scores share a lead-time axis.",
    )
    parser.add_argument(
        "--n-members", type=int, default=None,
        help="Ensemble members for checkpoints that sample stochastically "
             "(aifs-ens*). Each member is a fresh draw from the same initial "
             "state, so this multiplies runtime by n_members. Deterministic "
             "models reject values > 1 rather than returning identical copies.",
    )
    parser.add_argument(
        "--variables", nargs="+", default=["t2m"],
        help="One or more variables, e.g. t2m z500 t850 msl. Each is scored "
             "and reported separately. Variables a model does not predict "
             "are skipped for that model with a log line -- coverage "
             "differs (only AIFS and Aurora 1.5 produce precipitation).",
    )
    parser.add_argument(
        "--region", default="global",
        help="Restrict scoring to a region: a name (global, europe, conus, "
             "tropics, arctic, antarctic, north-atlantic, nh-extratropics, "
             "sh-extratropics) or 'lat_min,lat_max,lon_min,lon_max' with "
             "longitudes in -180..180. Results are written under a separate "
             "tag so they never mix with global scores.",
    )
    parser.add_argument(
        "--bin-regions", action="store_true",
        help="Score every region in one pass (tropics, extratropics, "
             "arctic, antarctic, europe, north-america) instead of only "
             "globally. Results gain a 'region' dimension and are written "
             "under a '#binned' tag.",
    )
    parser.add_argument(
        "--climatology", default=None,
        help="Path/URL to a climatology zarr. Required for ACC and the "
             "activity diagnostic; without it those metrics are skipped.",
    )
    parser.add_argument(
        "--score-stride-hours", type=int, default=6,
        help="Score only lead times that are multiples of this many hours. "
             "Aurora 1.5 emits hourly lead times while the AIFS models emit "
             "6-hourly; 6 puts them on a common axis. Set to 1 to score every "
             "step a model produces (then models are NOT comparable).",
    )
    parser.add_argument(
        "--source", default="era5", choices=["era5", "ifs"],
        help="Initial conditions AND verification truth. era5: historical and "
             "reproducible but lags ~a quarter. ifs: rolling ~2-week window, "
             "and the analysis AIFS/Aurora1.5 were fine-tuned on -- recent "
             "dates only, and not reproducible once the window moves.",
    )
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
        print("Metrics come from WeatherBench-X; see "
              "weather_pipeline.wbx_scoring.build_metrics for the set.")
        return

    # Same source for truth as for initial conditions, so a run is
    # self-consistent. Scoring one analysis against another is a separate
    # (deliberate) experiment, not something to do by accident.
    data_source = build_data_source(args.source)

    failed = []
    for init_time in args.init_times:
        for model in args.models:
            out_zarr = forecast_path(model, init_time)
            label = f"{model} @ {init_time}"
            # --n-steps is in benchmark (6 h) steps; each model runs the
            # number of its OWN steps that reaches the same lead time.
            steps = native_steps(model, args.n_steps)
            try:
                members = args.n_members if model in STOCHASTIC_MODELS else None
                if not args.force and existing_forecast_matches(
                    model, init_time, steps, out_zarr, members, args.source
                ):
                    logger.info("Reusing existing forecast for %s (use --force to re-run)", label)
                else:
                    run_model_isolated(model, init_time, steps, out_zarr,
                                       members, args.source)
            except subprocess.CalledProcessError:
                logger.exception("%s failed inside its own env -- skipping its scoring", label)
                failed.append(label)
                continue

            try:
                score_case(model, init_time, out_zarr, args.variables,
                           data_source, args.metrics, args.score_stride_hours,
                           args.region, args.bin_regions, args.climatology)
            except Exception:
                logger.exception("%s produced a forecast but scoring failed", label)
                failed.append(f"{label} (scoring)")

    if failed:
        logger.warning("Finished with failures: %s", failed)
    else:
        logger.info(
            "Complete: %d model(s) x %d init time(s). Build the comparison with:\n"
            "    uv run scripts/make_report.py --variable %s",
            len(args.models), len(args.init_times), args.variables[0],
        )


if __name__ == "__main__":
    main()