"""
Scoring via WeatherBench-X.

Replaces the hand-rolled metric registry in scoring.py. The point is
comparability: WB-X is the successor to the WeatherBench 2 evaluation code,
and the WB2 leaderboard is now computed with it -- so numbers produced here
can sit alongside published ones instead of being "our RMSE, probably the
same as theirs".

## What this buys over the old registry

  * metrics we could not compute before: ACC (needs a climatology), SEEPS,
    rank histograms, energy/variogram scores
  * correct aggregation for free. WB-X splits every metric into
    *statistics* (linearly averageable) and a final combination step, which
    is exactly the accumulator/finalise split scoring.py implemented by
    hand -- and it applies to metrics like spread-skill ratio that our
    version had to flag as "not accumulable".
  * regional and land/sea breakdowns via binning, with no new code
  * sparse targets (station observations), the same metrics applied to a
    truth source no model was trained on

## Schema

WB-X works in ECMWF conventions: forecasts indexed by (init_time,
lead_time), analyses by valid_time. The raw zarr our backends write is
already (time, lead_time, lat, lon), so the only change is renaming time ->
init_time; lat/lon are converted automatically by the loaders.

Note this is the RAW zarr, not what postprocess.load_forecast returns --
that collapses (time, lead_time) into a single valid_time axis, which is
the opposite of what WB-X wants. The plotting and notebook code still uses
the collapsed form; scoring no longer does.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Our forecasts name the ensemble dimension "ensemble"; WB-X defaults to
# ECMWF's "number".
ENSEMBLE_DIM = "ensemble"

# Metrics that consume the members themselves. Everything else is scored
# against the ensemble mean for an ensemble forecast.
ENSEMBLE_METRIC_NAMES = frozenset(
    {"crps", "ens_mean_rmse", "spread", "spread_skill_ratio"}
)


def build_metrics(
    variables: list[str] | None = None,
    climatology: xr.Dataset | None = None,
    ensemble: bool = False,
) -> dict:
    """The metric set to compute.

    Metrics that need a climatology (ACC, activity) are included only when
    one is supplied, and ensemble metrics only for ensemble forecasts --
    so one call site can score a mixed set of models without special
    casing."""
    from weatherbenchX.metrics import deterministic as det
    from weatherbenchX.metrics import probabilistic as prob

    metrics: dict = {
        "rmse": det.RMSE(),
        "mse": det.MSE(),
        "mae": det.MAE(),
        "bias": det.Bias(),
    }

    if climatology is not None:
        # ACC is the field WMO/ECMWF scorecards actually compare on, and
        # the 0.6 threshold for "useful synoptic skill" is defined against
        # it. PredictionActivity is WB-X's version of the over-smoothing
        # diagnostic: AI models blur with lead time, which flatters RMSE.
        metrics["acc"] = det.ACC(climatology=climatology)
        metrics["activity"] = det.PredictionActivity(climatology=climatology)

    # Wind is a VECTOR: scoring speed would need sqrt(u^2+v^2) computed on
    # each model's native grid before regridding, which scoring cannot do.
    # WindVectorRMSE scores the vector error from the components directly
    # and sidesteps that entirely.
    if variables and {"u10m", "v10m"} <= set(variables):
        metrics["wind_vector_rmse"] = det.WindVectorRMSE(
            u_name="u10m", v_name="v10m", vector_name="wind10m"
        )

    if ensemble:
        # fair=True is the ensemble-size-debiased CRPS, so ensembles with
        # different member counts stay comparable; the biased estimator
        # rewards small ensembles.
        metrics["crps"] = prob.CRPSEnsemble(ensemble_dim=ENSEMBLE_DIM, fair=True)
        metrics["ens_mean_rmse"] = prob.UnbiasedEnsembleMeanRMSE(ensemble_dim=ENSEMBLE_DIM)
        metrics["spread"] = prob.EnsembleRootMeanVariance(ensemble_dim=ENSEMBLE_DIM)
        # Includes the sqrt((K+1)/K) finite-ensemble correction, which the
        # raw spread/RMSE ratio omits.
        metrics["spread_skill_ratio"] = prob.UnbiasedSpreadSkillRatio(
            ensemble_dim=ENSEMBLE_DIM
        )

    return metrics


def open_forecast(zarr_path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open a forecast zarr in WB-X's expected layout.

    Deliberately NOT postprocess.load_forecast: that collapses
    (time, lead_time) into valid_time, and WB-X needs them separate.

    Renames are done here rather than leaning on the WB-X loaders'
    automatic lat/lon handling, because score_case hands DataArrays
    straight to the metrics and never goes through a loader."""
    ds = xr.open_zarr(zarr_path)
    for dim in ("batch",):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
    if variables is not None:
        ds = ds[variables]

    rename = {}
    if "time" in ds.dims or "time" in ds.coords:
        rename["time"] = "init_time"
    if "lat" in ds.dims or "lat" in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename["lon"] = "longitude"
    return ds.rename(rename)


def fetch_targets(source, forecast: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Analysis truth shaped like the forecast: (init_time, lead_time, lat, lon).

    Goes through ArraylakeERA5Source rather than WB-X's TargetsFromXarray
    because that source is where the alias tables, derived variables and
    grid standardisation live -- "z500" and "sic" are not variables in the
    repo, they are things the source knows how to resolve. Handing WB-X the
    raw store would lose all of that.

    The result is broadcast against the forecast's own coordinates, so the
    metrics see matching shapes with no alignment step."""
    init = np.asarray(forecast["init_time"].values)
    lead = np.asarray(forecast["lead_time"].values)
    valid = (init[:, None] + lead[None, :]).ravel()
    unique = pd.unique(pd.to_datetime(valid))
    logger.info("Fetching truth for %d unique valid time(s)", len(unique))

    da = source(list(unique), variables)          # (time, variable, lat, lon)
    da = da.rename({"lat": "latitude", "lon": "longitude"})
    da = _align_to_forecast_grid(da, forecast)

    out = {}
    idx = {pd.Timestamp(t): i for i, t in enumerate(unique)}
    order = np.array([idx[pd.Timestamp(t)] for t in valid])
    for v in variables:
        field = da.sel(variable=v, drop=True)
        vals = field.values[order].reshape(
            len(init), len(lead), field.sizes["latitude"], field.sizes["longitude"]
        )
        out[v] = (("init_time", "lead_time", "latitude", "longitude"), vals)

    return xr.Dataset(out, coords={
        "init_time": forecast["init_time"],
        "lead_time": forecast["lead_time"],
        "latitude": forecast["latitude"],
        "longitude": forecast["longitude"],
    })


def _align_to_forecast_grid(da: xr.DataArray, forecast: xr.Dataset) -> xr.DataArray:
    """Put truth on the forecast's grid.

    Models do not all output the same grid: ERA5 (and AIFS, after
    regridding) is 721 latitudes including both poles, while Aurora emits
    720 -- so truth and forecast differ by a row and the metrics cannot
    broadcast.

    Exact selection is used when the forecast's coordinates are a subset of
    truth's, which loses nothing. Otherwise linear interpolation, which
    does introduce error, so it is logged rather than done silently: for a
    fair benchmark every model should ideally be scored on ONE common grid
    (WeatherBench 2 regrids everything to 1.5 degrees for exactly this
    reason) rather than each on its own."""
    for dim in ("latitude", "longitude"):
        want = np.asarray(forecast[dim].values)
        have = np.asarray(da[dim].values)
        if len(want) == len(have) and np.allclose(want, have):
            continue
        if np.isin(want, have).all():
            logger.info("Truth %s: selecting %d of %d points to match the forecast",
                        dim, len(want), len(have))
            da = da.sel({dim: want})
        else:
            logger.warning(
                "Truth %s grid (%d points) does not contain the forecast's "
                "(%d points); interpolating. This adds interpolation error "
                "to this model's scores that models on the native grid do "
                "not pay.", dim, len(have), len(want),
            )
            da = da.interp({dim: want})
    return da


# Named regions as (lat_min, lat_max, lon_min, lon_max), longitudes in the
# -180..180 convention because that is how people write them; subset_region
# converts to the 0-360 the data is stored in.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (-90, 90, -180, 180),
    "europe": (35, 72, -25, 45),
    "north-atlantic": (20, 70, -80, 10),
    "conus": (24, 50, -125, -66),
    "tropics": (-20, 20, -180, 180),
    "nh-extratropics": (20, 90, -180, 180),
    "sh-extratropics": (-90, -20, -180, 180),
    "arctic": (60, 90, -180, 180),
    "antarctic": (-90, -60, -180, 180),
}


def parse_region(spec: str) -> tuple[float, float, float, float]:
    """A named region, or "lat_min,lat_max,lon_min,lon_max"."""
    if spec in REGIONS:
        return REGIONS[spec]
    try:
        lat0, lat1, lon0, lon1 = (float(x) for x in spec.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Region must be one of {sorted(REGIONS)} or four comma-separated "
            f"numbers 'lat_min,lat_max,lon_min,lon_max'; got {spec!r}"
        ) from exc
    return lat0, lat1, lon0, lon1


def subset_region(ds, region: tuple[float, float, float, float]):
    """Restrict a dataset to a lat/lon box.

    Boolean masks rather than slices, for two reasons: latitude is stored
    descending (so a slice would need reversed bounds), and a longitude
    range crossing the prime meridian is NOT contiguous in 0-360 storage --
    Europe at -25..45 becomes 335..360 plus 0..45. Masking handles the
    wrap; the resulting longitude coordinate is non-monotonic, which is
    fine here because the metrics reduce over it rather than interpolating.

    Area weighting still applies within the box, so a region spanning a
    wide latitude range is still weighted correctly.
    """
    lat0, lat1, lon0, lon1 = region
    lat = ds["latitude"].values
    lat_mask = (lat >= min(lat0, lat1)) & (lat <= max(lat0, lat1))

    lon = np.asarray(ds["longitude"].values) % 360
    a, b = lon0 % 360, lon1 % 360
    if np.isclose(a, b):                       # full circle
        lon_mask = np.ones_like(lon, dtype=bool)
    elif a <= b:
        lon_mask = (lon >= a) & (lon <= b)
    else:                                      # wraps through 0
        lon_mask = (lon >= a) | (lon <= b)

    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(f"Region {region} selects no grid points")

    return ds.isel(latitude=np.flatnonzero(lat_mask),
                   longitude=np.flatnonzero(lon_mask))


def mean_bias_field(forecast: xr.Dataset, targets: xr.Dataset, variable: str) -> xr.DataArray:
    """Time-mean (forecast - truth) map, for the report's spatial panels.

    Computed from the same arrays the metrics use, rather than re-loading
    the forecast in the collapsed valid_time layout: one read, one truth
    fetch, and no chance of the map and the scores disagreeing about which
    lead times or region they cover.

    Averaged over init_time and lead_time so a single small 2-D field is
    cached per case; an ensemble is collapsed to its mean first."""
    fc = forecast[variable]
    if has_ensemble(forecast):
        fc = fc.mean(dim=ENSEMBLE_DIM)
    diff = fc - targets[variable]
    out = diff.mean(dim=[d for d in ("init_time", "lead_time") if d in diff.dims])
    out.name = f"{variable}_mean_bias"
    return out


# Regions for binning, as {name: ((lat_min, lat_max), (lon_min, lon_max))}.
# Longitudes in 0-360, matching how the data is stored.
#
# Binning differs from subset_region above: it computes EVERY bin in one
# pass and adds a `region` dimension to the results, rather than throwing
# away the rest of the globe. So one run answers "how does this model do in
# the tropics vs the Arctic" instead of needing a run per region.
BIN_REGIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "tropics": ((-20, 20), (0, 360)),
    "nh-extratropics": ((20, 90), (0, 360)),
    "sh-extratropics": ((-90, -20), (0, 360)),
    "arctic": ((60, 90), (0, 360)),
    "antarctic": ((-90, -60), (0, 360)),
    "europe": ((35, 72), (335, 45)),
    "north-america": ((24, 60), (230, 300)),
}


def build_binning(
    regions: bool = True,
    latitude_bins: float | None = None,
    land_sea_mask: xr.DataArray | None = None,
):
    """Binning specs to pass to the aggregator.

    Returns None when nothing is requested, which the aggregator treats as
    "global only".

    Land/sea is the one that needs data: a land-sea fraction field on the
    same grid as the statistics. The ERA5 repo carries `lsm` as a
    coordinate, so it is available -- see weather_pipeline.statics."""
    from weatherbenchX import binning as wb_binning

    bins = []
    if regions:
        bins.append(wb_binning.Regions(BIN_REGIONS, land_sea_mask=land_sea_mask))
    if latitude_bins:
        bins.append(wb_binning.LatitudeBins(degrees=latitude_bins))
    if land_sea_mask is not None and not regions:
        bins.append(wb_binning.LandSea(land_sea_fraction=land_sea_mask))
    return bins or None


def load_climatology(path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open a climatology for ACC and the activity diagnostic.

    ACC measures skill at predicting the DEPARTURE from climatology, which
    is the field ECMWF scorecards compare on and where the "below 0.6 means
    no useful synoptic skill" rule of thumb is defined. It cannot be
    computed without one.

    Expected layout is WB2's: indexed by dayofyear and hour, so it can be
    matched to any valid time. WeatherBench publish one on GCS, or build
    your own from the ERA5 archive."""
    ds = xr.open_zarr(path)
    if variables is not None:
        ds = ds[[v for v in variables if v in ds]]
    rename = {}
    if "lat" in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims:
        rename["lon"] = "longitude"
    return ds.rename(rename) if rename else ds


def has_ensemble(ds: xr.Dataset) -> bool:
    return ENSEMBLE_DIM in ds.dims


def score_case(
    forecast: xr.Dataset,
    targets: xr.Dataset,
    variables: list[str],
    climatology: xr.Dataset | None = None,
    reduce_dims: tuple[str, ...] = ("init_time", "latitude", "longitude"),
    bin_by=None,
):
    """Score one forecast, returning (AggregationState, Dataset, metrics).

    The AggregationState is the thing to keep when scoring many init times:
    states SUM across cases, and only then are metric values computed.
    That ordering is what makes multi-case RMSE correct -- averaging
    finalised RMSEs gives a different, wrong number.
    """
    from weatherbenchX import aggregation, weighting
    from weatherbenchX.metrics import base as metrics_base

    ensemble = has_ensemble(forecast)
    metrics = build_metrics(variables, climatology, ensemble=ensemble)
    target_arrays = {v: targets[v] for v in variables}

    if not ensemble:
        stats = metrics_base.compute_unique_statistics_for_all_metrics(
            metrics, {v: forecast[v] for v in variables}, target_arrays
        )
    else:
        # Deterministic metrics must see the ENSEMBLE MEAN, not the members.
        # Handed the full array they broadcast over the ensemble dimension
        # and return one RMSE per member, which is both wrong (the standard
        # convention scores the mean) and leaves a stray `ensemble`
        # dimension in the results. Ensemble metrics get the members.
        det_metrics = {k: m for k, m in metrics.items()
                       if k not in ENSEMBLE_METRIC_NAMES}
        ens_metrics = {k: m for k, m in metrics.items()
                       if k in ENSEMBLE_METRIC_NAMES}
        mean_pred = {v: forecast[v].mean(dim=ENSEMBLE_DIM) for v in variables}
        full_pred = {v: forecast[v] for v in variables}

        stats = dict(metrics_base.compute_unique_statistics_for_all_metrics(
            det_metrics, mean_pred, target_arrays))
        stats.update(metrics_base.compute_unique_statistics_for_all_metrics(
            ens_metrics, full_pred, target_arrays))

    # GridAreaWeighting is WB-X's area weighting: the integral of cos over
    # each grid cell, so a polar cell counts for less than an equatorial
    # one. (Slightly more exact than point cos(lat), which the previous
    # hand-rolled scoring used -- the two agree to ~1e-5.)
    aggregator = aggregation.Aggregator(
        reduce_dims=list(reduce_dims),
        bin_by=bin_by,
        weigh_by=[weighting.GridAreaWeighting()],
        skipna=False,
    )
    state = aggregator.aggregate_statistics(stats)
    return state, state.metric_values(metrics), metrics


def combine_states(states: list, metrics) -> xr.Dataset:
    """Sum AggregationStates across cases, then compute metric values once.

    This is the correct multi-case aggregation: sum the linear statistics
    over every init time and finalise at the end."""
    from weatherbenchX import aggregation

    total = aggregation.AggregationState.sum(states)
    return total.metric_values(metrics)


def save_state(state, path, ensemble: bool = False) -> None:
    """Persist an AggregationState for later multi-case combination.

    Stored as a Dataset rather than finalised metric values, because
    combining cases means SUMMING states and computing metrics once at the
    end -- storing finished RMSEs would make correct aggregation
    impossible after the fact.

    The `ensemble` flag is recorded in the attrs: metric_values needs the
    SAME metric objects the state was built with, and inferring that from
    the stored statistic names is brittle."""
    ds = state.to_dataset()
    ds.attrs["ensemble"] = int(ensemble)
    ds.to_zarr(path, mode="w")


def load_state(path) -> tuple:
    """-> (AggregationState, was_ensemble)."""
    from weatherbenchX import aggregation

    ds = xr.open_zarr(path).load()
    ensemble = bool(ds.attrs.get("ensemble", 0))
    return aggregation.AggregationState.from_dataset(ds), ensemble


def to_dataframe(results: xr.Dataset) -> pd.DataFrame:
    """Metric values -> a lead-time-indexed DataFrame, matching the shape
    the report generator already consumes."""
    df = results.to_dataframe()
    if "lead_time" in df.index.names:
        df = df.reset_index()
        df["lead_time_h"] = (
            pd.to_timedelta(df["lead_time"]).dt.total_seconds() / 3600
        ).astype(int)
        df = df.set_index("lead_time_h").drop(columns=["lead_time"])
    return df