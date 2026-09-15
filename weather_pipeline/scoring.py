"""
On-demand scoring against ERA5 truth.

Metrics live in a REGISTRY (same shape as backends.BACKEND_FACTORIES) so
adding one is a single entry rather than a new function threaded through
the CLI. Select with a list of names or "all".

## The aggregation trap this is built to avoid

RMSE does not average. Scoring 30 init times and taking mean(RMSE) gives a
different (wrong) answer from the RMSE over all 30 cases. The same applies
to any metric with a non-linear final step.

So every metric is split in two:

    inner_fn   -> a quantity that IS linearly averageable across cases
                  (for RMSE: the weighted mean squared error)
    finalize   -> the non-linear last step (for RMSE: sqrt)

`compute_metrics` finalises immediately, which is what you want for a
single forecast. `compute_accumulators` returns the raw inner values, so a
multi-case benchmark can average those across init times and call
`finalize_accumulators` exactly once at the end. Metrics whose inner value
cannot be averaged across cases (ratios of variances, for instance) declare
`accumulable=False` and are refused by the accumulator path rather than
quietly producing a wrong number.

## Ensembles

Forecasts may carry an ensemble dimension. Deterministic metrics are
applied to the ensemble MEAN (the standard convention); ensemble metrics
(CRPS, spread) require the dimension and are skipped when it's absent.
"all" resolves to whichever metrics fit the forecast you actually have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:  # import only for type checking: keeps the metric
    # registry importable without arraylake, so report generation (and CI)
    # needs no data-access credentials or dependencies.
    from .datasource import ArraylakeERA5Source

logger = logging.getLogger(__name__)

ENSEMBLE_DIMS = ("ensemble", "member", "number")
SPATIAL_DIMS = ("latitude", "longitude")


def fetch_truth(
    source: "ArraylakeERA5Source", variable: str, valid_times
) -> xr.Dataset:
    """Pulls ERA5 analysis for one variable at the given valid times, in the
    same (valid_time, latitude, longitude) schema as `postprocess.load_forecast`."""
    logger.info("Fetching ERA5 truth for '%s' at %d valid times", variable, len(valid_times))
    da = source(valid_times, [variable])  # dims: time, variable, lat, lon
    da = da.sel(variable=variable, drop=True)
    da = da.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})
    return da.to_dataset(name=variable)


def _latitude_weights(latitude: xr.DataArray) -> xr.DataArray:
    """cos(lat), so a polar grid cell doesn't count as much as an equatorial
    one. Deliberately not normalised: xarray's .weighted().mean() divides by
    the weight sum itself, so any constant factor cancels."""
    return np.cos(np.deg2rad(latitude))


def ensemble_dim(da: xr.DataArray) -> str | None:
    return next((d for d in ENSEMBLE_DIMS if d in da.dims), None)


def _wmean(da: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    return da.weighted(weights).mean(dim=SPATIAL_DIMS)


# ---------------------------------------------------------------------------
# Metric implementations: (fc, truth, weights) -> one value per lead time.
# `fc` still carries its ensemble dim for metrics that asked to keep it.
# ---------------------------------------------------------------------------

def _inner_mse(fc, truth, weights):
    return _wmean((fc - truth) ** 2, weights)


def _inner_bias(fc, truth, weights):
    return _wmean(fc - truth, weights)


def _inner_mae(fc, truth, weights):
    return _wmean(abs(fc - truth), weights)


def _inner_extreme_mse(fc, truth, weights):
    """MSE restricted to points where truth is in its own top 1%.

    Bulk RMSE is dominated by the bulk, so a model can score well while
    missing every extreme. The threshold comes from the truth field per
    lead time, so it adapts instead of needing a hand-set physical value
    per variable."""
    thresh = truth.quantile(0.99, dim=SPATIAL_DIMS)
    mask = truth >= thresh
    sq = ((fc - truth) ** 2).where(mask)
    return sq.weighted((weights * mask).fillna(0)).mean(dim=SPATIAL_DIMS)


def _activity_ratio(fc, truth, weights):
    """Forecast spatial variance / truth spatial variance.

    AI models blur with lead time, and blurring IMPROVES RMSE while making
    the field less physical -- this is the diagnostic that catches it.
    Below 1 means over-smoothed. Not accumulable: a ratio of per-case
    variances cannot be averaged across cases."""
    fc_var = fc.weighted(weights).var(dim=SPATIAL_DIMS)
    truth_var = truth.weighted(weights).var(dim=SPATIAL_DIMS)
    return fc_var / truth_var


def _crps(fc, truth, weights):
    """Fair (ensemble-size-debiased) CRPS.

        CRPS = E|X - y| - 0.5 * E|X - X'|

    The second term uses 1/(m(m-1)) rather than 1/m^2 so ensembles of
    different sizes stay comparable -- the biased estimator rewards small
    ensembles. Reduces to MAE at m=1, which puts deterministic and ensemble
    runs on one axis."""
    dim = ensemble_dim(fc)
    m = fc.sizes[dim]
    skill = abs(fc - truth).mean(dim=dim)
    if m == 1:
        spread_term = xr.zeros_like(skill)
    else:
        a = fc.rename({dim: "_i"})
        b = fc.rename({dim: "_j"})
        spread_term = abs(a - b).sum(dim=("_i", "_j")) / (2 * m * (m - 1))
    return _wmean(skill - spread_term, weights)


def _ensemble_variance(fc, truth, weights):
    """Mean ensemble variance; finalised to a standard deviation.

    Compare against ensemble-mean RMSE -- a calibrated ensemble has spread
    close to skill, and AI ensembles are classically under-dispersed."""
    dim = ensemble_dim(fc)
    return _wmean(fc.var(dim=dim, ddof=1), weights)


@dataclass(frozen=True)
class Metric:
    name: str
    inner_fn: Callable
    finalize: Callable = lambda x: x
    needs_ensemble: bool = False
    # Deterministic metrics collapse the ensemble dim by taking its mean
    # before scoring; ensemble metrics need the members themselves.
    use_ensemble_mean: bool = True
    # False where the inner quantity cannot be averaged across init times.
    accumulable: bool = True
    description: str = ""


REGISTRY: dict[str, Metric] = {
    "rmse": Metric(
        "rmse", _inner_mse, np.sqrt,
        description="Latitude-weighted RMSE. Inner quantity is MSE, so it "
                    "aggregates correctly across cases.",
    ),
    "mse": Metric("mse", _inner_mse, description="Latitude-weighted mean squared error."),
    "bias": Metric(
        "bias", _inner_bias,
        description="Mean error (forecast - truth). RMSE hides systematic "
                    "drift; a uniformly warm model can still score well.",
    ),
    "mae": Metric(
        "mae", _inner_mae,
        description="Mean absolute error. Read alongside RMSE: a large "
                    "RMSE/MAE gap means a few big errors, not many small.",
    ),
    "extreme_rmse_p99": Metric(
        "extreme_rmse_p99", _inner_extreme_mse, np.sqrt,
        description="RMSE over points where truth exceeds its own 99th "
                    "percentile -- tail skill, which bulk RMSE hides.",
    ),
    "activity_ratio": Metric(
        "activity_ratio", _activity_ratio, accumulable=False,
        description="Forecast variance / truth variance. <1 means the model "
                    "is over-smoothing, which flatters its RMSE.",
    ),
    "crps": Metric(
        "crps", _crps, needs_ensemble=True, use_ensemble_mean=False,
        description="Fair CRPS. The standard proper score for ensembles; "
                    "equals MAE for a single member.",
    ),
    "spread": Metric(
        "spread", _ensemble_variance, np.sqrt,
        needs_ensemble=True, use_ensemble_mean=False,
        description="Mean ensemble standard deviation. Compare with RMSE of "
                    "the ensemble mean: spread ~ skill means calibrated.",
    ),
}


def describe_metrics() -> str:
    """One line per metric, for `--list-metrics` and docs."""
    return "\n".join(
        f"  {m.name:18s} {'[ens] ' if m.needs_ensemble else '      '}{m.description}"
        for m in REGISTRY.values()
    )


def resolve_metrics(names, has_ensemble: bool) -> list[Metric]:
    """Turn a selection into Metric objects.

    "all" means every metric that APPLIES to this forecast -- ensemble
    metrics are dropped for a deterministic run rather than erroring, so one
    CLI invocation can score a mixed set of models. Naming an ensemble
    metric explicitly for a deterministic forecast IS an error, because that
    request cannot be honoured and silently dropping it would mislead."""
    if names in ("all", ["all"], None):
        chosen = [m for m in REGISTRY.values() if has_ensemble or not m.needs_ensemble]
        logger.info("Metrics: %s", [m.name for m in chosen])
        return chosen

    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]

    out = []
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"Unknown metric '{n}'. Available: {sorted(REGISTRY)} (or 'all')")
        m = REGISTRY[n]
        if m.needs_ensemble and not has_ensemble:
            raise ValueError(
                f"Metric '{n}' needs an ensemble dimension, but this forecast "
                f"has none (looked for {ENSEMBLE_DIMS}). Ensemble members "
                "require perturbed initial conditions -- see the note in "
                "anemoi_backend.AnemoiBackend.run."
            )
        out.append(m)
    return out


def _prepare(forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str):
    fc = forecast_ds[variable]
    truth = truth_ds[variable].sel(valid_time=fc["valid_time"])
    return fc, truth, _latitude_weights(fc["latitude"])


def _lead_index(fc: xr.DataArray) -> pd.Index:
    hours = (fc["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    return pd.Index(hours, name="lead_time_h")


def compute_accumulators(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str, metrics="all"
) -> pd.DataFrame:
    """Per-lead-time INNER values, before any non-linear finalisation.

    This is what to accumulate when scoring many init times: average these
    across cases, then call `finalize_accumulators` once at the end.
    Averaging finalised RMSE across cases gives a different, wrong answer."""
    fc, truth, weights = _prepare(forecast_ds, truth_ds, variable)
    ens = ensemble_dim(fc)
    chosen = resolve_metrics(metrics, has_ensemble=ens is not None)

    cols = {}
    for m in chosen:
        if not m.accumulable:
            logger.info("Skipping '%s' in accumulator mode (not averageable across cases)", m.name)
            continue
        f = fc.mean(dim=ens) if (ens and m.use_ensemble_mean) else fc
        cols[m.name] = np.asarray(m.inner_fn(f, truth, weights).values, dtype=float)
    return pd.DataFrame(cols, index=_lead_index(fc))


def finalize_accumulators(acc: pd.DataFrame) -> pd.DataFrame:
    """Apply each metric's final non-linear step to averaged inner values."""
    return pd.DataFrame(
        {name: REGISTRY[name].finalize(acc[name]) for name in acc.columns},
        index=acc.index,
    )


def compute_metrics(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str, metrics="all"
) -> pd.DataFrame:
    """Finalised metrics for ONE forecast, indexed by lead time (hours)."""
    fc, truth, weights = _prepare(forecast_ds, truth_ds, variable)
    ens = ensemble_dim(fc)
    if ens:
        logger.info("Forecast has %d ensemble member(s) on dim '%s'", fc.sizes[ens], ens)
    chosen = resolve_metrics(metrics, has_ensemble=ens is not None)

    cols = {}
    for m in chosen:
        f = fc.mean(dim=ens) if (ens and m.use_ensemble_mean) else fc
        cols[m.name] = m.finalize(np.asarray(m.inner_fn(f, truth, weights).values, dtype=float))
    out = pd.DataFrame(cols, index=_lead_index(fc))
    logger.info("Computed %s for '%s' at %d lead times", list(out.columns), variable, len(out))
    return out


def score_case(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str, metrics="all"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score ONE forecast, returning (accumulators, finalized).

    Both, because they serve different purposes: `finalized` is what you
    read for this single case, while `accumulators` is what a multi-case
    benchmark must average before finalising. Writing both to disk lets a
    report aggregate correctly without re-reading the forecasts."""
    acc = compute_accumulators(forecast_ds, truth_ds, variable, metrics)
    fin = compute_metrics(forecast_ds, truth_ds, variable, metrics)
    return acc, fin


def mean_bias_field(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str
) -> xr.DataArray:
    """Time-mean (forecast - truth) map, for the spatial panels in the
    report. Averaged over lead time so one small 2-D field per model can be
    cached instead of the whole error cube."""
    diff = compute_difference(forecast_ds, truth_ds, variable)
    out = diff.mean(dim="valid_time")
    out.name = f"{variable}_mean_bias"
    return out


def compute_rmse(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str
) -> pd.Series:
    """Latitude-weighted RMSE for one variable, indexed by lead time (hours).

    Kept for the notebook and plotting.plot_rmse; equivalent to
    compute_metrics(..., metrics=["rmse"])."""
    s = compute_metrics(forecast_ds, truth_ds, variable, metrics=["rmse"])["rmse"]
    s.name = f"rmse_{variable}"
    return s


def compute_difference(
    forecast_ds: xr.Dataset, truth_ds: xr.Dataset, variable: str
) -> xr.DataArray:
    """Forecast-minus-truth error field, one value per grid point per lead
    time -- i.e. the spatial map RMSE is a summary of, rather than the
    reduced RMSE number itself.

    Keeps the full (valid_time, latitude, longitude) resolution, so you can
    feed it straight into `plotting.render_rotating_gif(..., center=0)` to
    animate where and how the error grows with lead time. An ensemble is
    collapsed to its mean first."""
    logger.info("Computing forecast-minus-truth difference for '%s'", variable)
    fc = forecast_ds[variable]
    ens = ensemble_dim(fc)
    if ens:
        fc = fc.mean(dim=ens)
    truth = truth_ds[variable].sel(valid_time=fc["valid_time"])
    diff = fc - truth
    diff.name = f"{variable}_error"
    return diff