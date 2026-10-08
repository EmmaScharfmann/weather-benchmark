"""
Scoring downscaled forecasts against station observations, with
WeatherBench-X doing the arithmetic.

Why WB-X and not our own: it is the evaluation code behind the WeatherBench 2
leaderboard, and it already has what station data needs -- statistics that
sum correctly across cases, binning by any coordinate (elevation, region),
and StationDensityWeighting, which stops a dense cluster of stations from
counting for more than an isolated one. This module is glue: our (init_time,
lead_time, [ensemble], station) arrays in, WB-X metrics out.

Typical use, once per initialisation, then combined:

    cases = []
    for zarr in forecasts:
        fc_st = downscale_to_stations(open_zarr(zarr), stations, "t2m", orography=oro)
        obs   = observations_at_valid_times(fc_st, stations, "t2m")
        an_st = analysis_baseline(era5, open_zarr(zarr), stations, "t2m", orography=oro)
        cases.append(score_station_case(fc_st, obs, analysis=an_st))
    results = combine_cases(cases)            # {method: Dataset(lead_time, region, elevation)}
    lo, hi  = bootstrap_ci(cases, "bilinear_lapse")

Combining works by SUMMING per-case statistics and only then taking the
square root for RMSE. Averaging per-case RMSEs gives a different, wrong
number, which is why cases are kept as states rather than finished scores.

Not here yet: threshold-exceedance (frost / heat event) scores. They need a
decision about station-specific thresholds first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

ENSEMBLE_DIM = "ensemble"
ENSEMBLE_METRICS = frozenset({"crps", "ens_mean_rmse", "spread", "ssr"})

# Approximate boxes ((lat_min, lat_max), (lon_min, lon_max)), signed lon.
# They are ROUGH and meant to be reviewed: they overlap on purpose (a
# station can sit in more than one) and are chosen to separate terrain
# regimes, not to follow administrative borders.
FRANCE_REGIONS = {
    "france": ((41.0, 51.5), (-5.5, 10.0)),
    "alps": ((44.0, 46.8), (5.8, 7.8)),
    "pyrenees": ((42.2, 43.3), (-1.8, 3.0)),
    "massif-central": ((44.5, 46.0), (2.0, 4.0)),
    "mediterranean": ((42.5, 44.0), (3.0, 7.5)),
    "atlantic": ((43.5, 48.5), (-5.0, -0.5)),
    "north": ((48.5, 51.0), (-4.5, 8.0)),
}
ELEVATION_EDGES = np.array([0, 200, 500, 1000, 1500, 2500, 5000])


def build_metrics(ensemble: bool) -> dict:
    """Metric set. Ensemble metrics only for ensemble forecasts, so a mixed
    ladder (ensemble forecast vs deterministic analysis baseline) can be
    scored with one call site."""
    from weatherbenchX.metrics import deterministic as det
    from weatherbenchX.metrics import probabilistic as prob

    m = {"rmse": det.RMSE(), "mae": det.MAE(), "bias": det.Bias()}
    if ensemble:
        # fair=True: ensemble-size-debiased, so ensembles of different sizes
        # stay comparable (the plain estimator rewards small ensembles).
        m["crps"] = prob.CRPSEnsemble(ensemble_dim=ENSEMBLE_DIM, fair=True)
        m["ens_mean_rmse"] = prob.UnbiasedEnsembleMeanRMSE(ensemble_dim=ENSEMBLE_DIM)
        m["spread"] = prob.EnsembleRootMeanVariance(ensemble_dim=ENSEMBLE_DIM)
        m["ssr"] = prob.UnbiasedSpreadSkillRatio(ensemble_dim=ENSEMBLE_DIM)
    return m


def default_bins(regions: Mapping | None = FRANCE_REGIONS,
                 elevation_edges: Sequence[float] | None = ELEVATION_EDGES) -> list:
    """Region x elevation bins. Either may be None to drop it. WB-X multiplies
    the binnings, so the result has both a `region` and an `elevation` dim."""
    from weatherbenchX import binning
    bins = []
    if regions:
        bins.append(binning.Regions(dict(regions)))
    if elevation_edges is not None:
        bins.append(binning.ByCoordBins("elevation", np.asarray(elevation_edges)))
    return bins


def select_stations(*arrays, which: str = "all", holdout: xr.DataArray | np.ndarray | None = None):
    """Subset stations of any number of arrays. `which`: all | holdout | train.

    The baselines never see station data, so they can be scored on `all`.
    A learned downscaler must be scored on `holdout` only."""
    if which == "all":
        return arrays if len(arrays) > 1 else arrays[0]
    if holdout is None:
        raise ValueError("`holdout` mask required for which='holdout'/'train'")
    mask = np.asarray(holdout, dtype=bool)
    if which == "train":
        mask = ~mask
    elif which != "holdout":
        raise ValueError("which must be all, holdout or train")
    out = tuple(a.isel(station=np.flatnonzero(mask)) for a in arrays)
    return out if len(out) > 1 else out[0]


@dataclass
class StationScores:
    """Per-case accumulated statistics, one state per method. Keep these, not
    finished metrics: they are what combine correctly across cases."""
    states: dict
    ensemble: dict
    n_obs: dict = field(default_factory=dict)


def _scoring_inputs(da: xr.DataArray, like: xr.Dataset) -> xr.DataArray:
    """Strip 2-D valid_time coords (they cannot survive reduction over
    init_time) and make sure the station coords WB-X bins/weights on exist."""
    da = da.drop_vars([c for c in ("valid_time",) if c in da.coords])
    for c in ("latitude", "longitude", "elevation"):
        if c not in da.coords:
            da = da.assign_coords({c: ("station", like[c].values)})
    return da


def score_station_case(
    fc_st: xr.Dataset,
    obs: xr.DataArray,
    variable: str = "t2m",
    analysis: xr.Dataset | None = None,
    methods: Sequence[str] | None = None,
    bin_by: Sequence | None = "default",
    density_alpha_deg: float | None = 0.75,
) -> StationScores:
    """Score every method of one initialisation against `obs`.

    methods default to every of nearest / bilinear / bilinear_lapse present.
    `analysis` (from analysis_baseline) adds methods named analysis_<name>,
    scored deterministically -- the representativeness floor.
    density_alpha_deg=None switches density weighting off (plain mean over
    stations)."""
    from weatherbenchX import aggregation, weighting
    from weatherbenchX.metrics import base as metrics_base

    bins = default_bins() if isinstance(bin_by, str) else bin_by
    methods = list(methods or [m for m in ("nearest", "bilinear", "bilinear_lapse") if m in fc_st])
    obs = _scoring_inputs(obs, fc_st)
    weights = [weighting.StationDensityWeighting(alpha_0_degrees=density_alpha_deg)] \
        if density_alpha_deg else None
    # skipna: station gaps are expected, and WB-X then drops them from the
    # weights as well as the sums, so a gap does not bias the mean.
    agg = aggregation.Aggregator(reduce_dims=["init_time", "station"], bin_by=bins,
                                 weigh_by=weights, skipna=True)

    jobs = {m: (fc_st[m], m) for m in methods}
    if analysis is not None:
        jobs.update({f"analysis_{m}": (analysis[m], m) for m in ("nearest", "bilinear", "bilinear_lapse")
                     if m in analysis})

    states, ens_flag, n_obs = {}, {}, {}
    for name, (da, _) in jobs.items():
        da = _scoring_inputs(da, fc_st)
        # ensembles of different sizes cannot share one dimension name in a
        # Dataset (xarray would pad with NaN), so any dim named ensemble* counts
        ens_dims = [d for d in da.dims if str(d).startswith(ENSEMBLE_DIM)]
        if ens_dims and ens_dims[0] != ENSEMBLE_DIM:
            da = da.rename({ens_dims[0]: ENSEMBLE_DIM})
        if ENSEMBLE_DIM in da.coords:
            da = da.drop_vars(ENSEMBLE_DIM)
        is_ens = ENSEMBLE_DIM in da.dims
        metrics = build_metrics(is_ens)
        det_m = {k: v for k, v in metrics.items() if k not in ENSEMBLE_METRICS}
        stats = dict(metrics_base.compute_unique_statistics_for_all_metrics(
            det_m, {variable: da.mean(ENSEMBLE_DIM) if is_ens else da}, {variable: obs}))
        if is_ens:
            ens_m = {k: v for k, v in metrics.items() if k in ENSEMBLE_METRICS}
            stats.update(metrics_base.compute_unique_statistics_for_all_metrics(
                ens_m, {variable: da}, {variable: obs}))
        states[name] = agg.aggregate_statistics(stats)
        ens_flag[name] = is_ens
        n_obs[name] = int(np.isfinite(obs.values).sum())
    return StationScores(states, ens_flag, n_obs)


def _values(state, ensemble: bool) -> xr.Dataset:
    ds = state.metric_values(build_metrics(ensemble))
    # WB-X names results "<metric>.<variable>"; with one variable keep just the metric
    return ds.rename({k: k.split(".")[0] for k in ds.data_vars})


def combine_cases(cases: Sequence[StationScores]) -> dict[str, xr.Dataset]:
    """Sum statistics over cases, then compute metrics once, per method."""
    from weatherbenchX import aggregation
    out = {}
    for name in cases[0].states:
        total = aggregation.AggregationState.sum([c.states[name] for c in cases])
        out[name] = _values(total, cases[0].ensemble[name])
    return out


def bootstrap_ci(
    cases: Sequence[StationScores],
    method: str,
    n_boot: int = 300,
    block: int = 2,
    seed: int = 0,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Block bootstrap over cases (initialisations).

    Neighbouring inits share weather, so resampling them independently makes
    the intervals too narrow. Blocks of `block` consecutive cases are
    resampled with replacement; with 2-day init spacing, block=2 keeps 4-day
    persistence together. With about a dozen cases the intervals are
    ILLUSTRATIVE: they describe sampling noise in this one heatwave, not
    uncertainty about heatwaves in general."""
    from weatherbenchX import aggregation
    n = len(cases)
    if n < 3:
        raise ValueError("bootstrap needs at least 3 cases")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    ens = cases[0].ensemble[method]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:n]
        total = aggregation.AggregationState.sum([cases[i].states[method] for i in idx])
        draws.append(_values(total, ens))
    stacked = xr.concat(draws, dim="boot")
    lo, hi = (stacked.quantile(q, dim="boot", skipna=True).drop_vars("quantile") for q in quantiles)
    return lo, hi


def to_frame(results: Mapping[str, xr.Dataset], metric: str = "rmse", **sel) -> pd.DataFrame:
    """Methods side by side for one metric: rows = lead time (h), columns =
    methods. Extra keyword args select bin coordinates, e.g.
    region="alps", elevation=1000."""
    cols = {}
    for name, ds in results.items():
        da = ds[metric]
        if sel:
            da = da.sel(sel)
        da = da.squeeze(drop=True)
        if da.ndim != 1:
            raise ValueError(f"{name}: {metric} still has dims {da.dims}; select more bins")
        cols[name] = pd.Series(da.values, index=(da["lead_time"].values / np.timedelta64(1, "h")).astype(int))
    return pd.DataFrame(cols).rename_axis("lead_h")