"""
Station observations as verification truth for downscaled forecasts.

    ds, report = load_stations(MeteoFranceHourly(), "france",
                               "2023-08-05", "2023-08-30", variables=["t2m"])

`ds` has dims (time, station), one data variable per canonical id ("t2m" in
KELVIN, the same id and unit the forecast zarrs use), and per-station
coordinates: latitude, longitude, elevation, name. `report` is a per-station
QC table saying who was kept and why not.

Layout:
  * `StationProvider` -- the two things a national network has to supply
    (a station list, and long-format values). `MeteoFranceHourly` is the
    first implementation; another country is another class, nothing else.
  * `load_stations` -- provider-independent: filter to stations that were
    active, fetch only what the cache lacks, QC, assemble, tag the holdout.

Conventions:
  * times are naive UTC (datetime64[ns]); hourly, inclusive of both ends
  * station longitude is SIGNED (-180..180), as networks publish it. The
    forecast zarrs use 0..360 -- convert with regions.to_signed_lon before
    interpolating between the two
  * a station must be active over the WHOLE window to be considered, so
    completeness is meaningful (set active="overlap" to relax this)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Region, parse_region

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical variables: id -> (unit, plausible range in that unit). The
# range is a gross-error filter, not a climatology: it exists to catch
# sentinel values (-999, 9999) and unit mistakes, not to judge a heatwave.
# ---------------------------------------------------------------------------
VARIABLES: dict[str, dict] = {
    "t2m":  {"unit": "K", "range": (223.15, 333.15)},   # -50 .. 60 degC
    "tx1h": {"unit": "K", "range": (223.15, 333.15)},   # max within the hour
    "tn1h": {"unit": "K", "range": (223.15, 333.15)},   # min within the hour
}


class StationUnavailable(Exception):
    """A station has no data for the requested period (e.g. it closed).

    Distinct from a network error on purpose: this is a permanent fact and
    is cached as 'no data', whereas a network error must NOT be cached or the
    station would be silently missing from every later run."""


class StationProvider(Protocol):
    name: str
    license: str

    def list_stations(self, region: Region, start: pd.Timestamp,
                      end: pd.Timestamp, variables: Sequence[str]) -> pd.DataFrame:
        """Columns: station_id (str), name, latitude, longitude (signed),
        elevation (m), start_date, end_date (naive UTC; NaT = unknown/open)."""

    def fetch(self, station_ids: Sequence[str], start: pd.Timestamp,
              end: pd.Timestamp, variables: Sequence[str]) -> pd.DataFrame:
        """Long format: station_id, variable (canonical id), time (naive
        UTC), value (canonical unit). Raise StationUnavailable for a single
        station with no data; return no rows for it if called with several."""


# ---------------------------------------------------------------------------
# Meteo-France hourly network (through wetterdienst)
# ---------------------------------------------------------------------------
class MeteoFranceHourly:
    """Hourly 'core' observations, ~thousands of stations, Etalab Open
    License 2.0 (attribute: Meteo-France).

    Verified in the sandbox against wetterdienst 0.141.0: station list with
    elevation, and hourly values at HH:00 UTC. Not verified: the 'recent'
    vs 'historical' period boundary for 2024-25 data."""

    name = "meteofrance_hourly"
    license = "Etalab Open License 2.0 - Source: Meteo-France"

    # canonical id -> (resolution, dataset, wetterdienst parameter, offset)
    # offset converts the native unit (degC) to the canonical one (K)
    _PARAMS = {
        "t2m":  ("hourly", "core", "temperature_air_mean_2m", 273.15),
        "tx1h": ("hourly", "core", "temperature_air_max_2m", 273.15),
        "tn1h": ("hourly", "core", "temperature_air_min_2m", 273.15),
    }

    def __init__(self, chunk_size: int = 25, cache_disable: bool = False):
        self.chunk_size = chunk_size
        # wetterdienst keeps its own file cache (default ~/.cache/wetterdienst).
        # A half-written entry there (an interrupted download) surfaces as
        # "TypeError: cannot unpack non-iterable bool object" deep in fsspec.
        # Our own per-station cache makes it redundant, so it can be disabled;
        # fetch() also falls back to no-cache for a chunk that hits that error.
        self.cache_disable = cache_disable

    def _request(self, variables, start, end, cache_disable: bool | None = None):
        from wetterdienst import Settings
        from wetterdienst.provider.meteofrance.observation import (
            MeteoFranceObservationRequest,
        )
        unknown = [v for v in variables if v not in self._PARAMS]
        if unknown:
            raise KeyError(f"{unknown} not supported by {self.name}; "
                           f"available: {sorted(self._PARAMS)}")
        params = [self._PARAMS[v][:3] for v in variables]
        disable = self.cache_disable if cache_disable is None else cache_disable
        return MeteoFranceObservationRequest(
            parameters=params, start_date=start.isoformat(), end_date=end.isoformat(),
            settings=Settings(cache_disable=disable))

    @staticmethod
    def _naive_utc(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, utc=True)
        return s.dt.tz_convert("UTC").dt.tz_localize(None)

    def list_stations(self, region, start, end, variables):
        lat0, lat1, lon0, lon1 = region
        req = self._request(variables, start, end)
        df = req.filter_by_bbox(left=lon0, bottom=min(lat0, lat1),
                                right=lon1, top=max(lat0, lat1)).df.to_pandas()
        df = df.drop_duplicates("station_id").copy()
        for c in ("start_date", "end_date"):
            df[c] = self._naive_utc(df[c])
        df["station_id"] = df["station_id"].astype(str)
        return df[["station_id", "name", "latitude", "longitude", "elevation",
                   "start_date", "end_date"]].reset_index(drop=True)

    @staticmethod
    def _missing_file(exc: BaseException) -> bool:
        while exc is not None:
            if isinstance(exc, FileNotFoundError):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    @staticmethod
    def _broken_cache(exc: BaseException) -> bool:
        return isinstance(exc, TypeError) and "non-iterable bool" in str(exc)

    def _fetch_chunk(self, ids, start, end, variables, cache_disable: bool | None = None) -> pd.DataFrame:
        req = self._request(variables, start, end, cache_disable).filter_by_station_id(list(ids))
        try:
            df = req.values.all().df.to_pandas()
        except TypeError as exc:
            if cache_disable or not self._broken_cache(exc):
                raise
            logger.warning("wetterdienst file cache looks corrupted (%s); retrying this chunk without it. "
                           "Consider clearing its cache directory.", exc)
            return self._fetch_chunk(ids, start, end, variables, cache_disable=True)
        if df.empty:
            return pd.DataFrame(columns=["station_id", "variable", "time", "value"])
        by_native = {self._PARAMS[v][2]: v for v in variables}
        offset = {v: self._PARAMS[v][3] for v in variables}
        out = pd.DataFrame({
            "station_id": df["station_id"].astype(str),
            "variable": df["parameter"].astype(str).map(by_native),
            "time": self._naive_utc(df["timestamp"]),
            "value": df["value"].astype("f8"),
        }).dropna(subset=["variable"])
        out["value"] = out["value"] + out["variable"].map(offset)
        return out[(out["time"] >= start) & (out["time"] <= end)]

    def fetch(self, station_ids, start, end, variables):
        ids = [str(i) for i in station_ids]
        frames = []
        for i in range(0, len(ids), self.chunk_size):
            chunk = ids[i:i + self.chunk_size]
            try:
                frames.append(self._fetch_chunk(chunk, start, end, variables))
            except Exception as exc:  # noqa: BLE001
                if not self._missing_file(exc):
                    raise
                # one bad station poisons its whole chunk: retry singly so
                # the good ones still come through
                for sid in chunk:
                    try:
                        frames.append(self._fetch_chunk([sid], start, end, variables))
                    except Exception as exc2:  # noqa: BLE001
                        if not self._missing_file(exc2):
                            raise
                        logger.info("Station %s has no data file for the window", sid)
        return pd.concat(frames, ignore_index=True) if frames else \
            pd.DataFrame(columns=["station_id", "variable", "time", "value"])


# ---------------------------------------------------------------------------
# QC / holdout
# ---------------------------------------------------------------------------
@dataclass
class QCConfig:
    # Fraction of the window's hours a station needs a valid value for.
    min_completeness: float = 0.9
    # Variable used to judge completeness; others are only range-checked.
    primary: str = "t2m"
    # Gross-error ranges come from VARIABLES; override per variable here.
    ranges: dict | None = None


def holdout_mask(station_ids: Sequence[str], frac: float = 0.2, salt: str = "dkit-v1") -> np.ndarray:
    """Deterministic ~`frac` of stations, decided by a hash of the station id.

    Hash rather than random draw: the set is the same on every run and does
    not reshuffle when stations are added or dropped, so a model trained
    last month and scored today never sees yesterday's holdout in training."""
    out = np.empty(len(station_ids), dtype=bool)
    for i, sid in enumerate(station_ids):
        h = int(hashlib.md5(f"{salt}:{sid}".encode()).hexdigest()[:8], 16)
        out[i] = (h % 10_000) < frac * 10_000
    return out


# ---------------------------------------------------------------------------
# cache (one small csv.gz per station x variable x window; no extra deps)
# ---------------------------------------------------------------------------
def _cache_file(cache_dir, provider, var, start, end, sid) -> Path:
    tag = f"{start:%Y%m%dT%H}_{end:%Y%m%dT%H}"
    return Path(cache_dir) / provider / var / tag / f"{sid}.csv.gz"


_EMPTY = pd.DataFrame(columns=["station_id", "variable", "time", "value"])


def _read_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"], dtype={"station_id": str})


def _covering_dirs(cache_dir, provider, var, start, end) -> list[Path]:
    """Cached window folders for this variable whose window COVERS [start, end],
    smallest first. A cache filled for 5-30 Aug then serves a 6-20 Aug request
    instead of refetching every station."""
    root = Path(cache_dir) / provider / var
    out = []
    if root.is_dir():
        for d in root.iterdir():
            try:
                a, b = (pd.to_datetime(x, format="%Y%m%dT%H") for x in d.name.split("_"))
            except (ValueError, TypeError):
                continue
            if a <= start and b >= end.floor("h"):
                out.append((b - a, d))
    return [d for _, d in sorted(out)]


def _cached_file(cache_dir, provider, var, start, end, sid) -> Path | None:
    exact = _cache_file(cache_dir, provider, var, start, end, sid)
    if exact.exists():
        return exact
    for d in _covering_dirs(cache_dir, provider, var, start, end):
        if (d / f"{sid}.csv.gz").exists():
            return d / f"{sid}.csv.gz"
    return None


def _fetch_with_cache(provider, ids, start, end, variables, cache_dir) -> pd.DataFrame:
    need, frames = [], []
    for sid in ids:
        if any(_cached_file(cache_dir, provider.name, v, start, end, sid) is None for v in variables):
            need.append(sid)
    hits = len(ids) - len(need)
    logger.info("Station cache: %d/%d station(s) complete, fetching %d",
                hits, len(ids), len(need))
    if need and hits == 0:
        root = Path(cache_dir) / provider.name / variables[0]
        if root.is_dir():
            logger.info("Cached windows present for %s: %s (none covers %s .. %s)", variables[0],
                        sorted(d.name for d in root.iterdir()), start, end)

    if need:
        got = provider.fetch(need, start, end, variables)
        for sid in need:
            sub = got[got["station_id"] == sid] if len(got) else _EMPTY
            for v in variables:
                p = _cache_file(cache_dir, provider.name, v, start, end, sid)
                p.parent.mkdir(parents=True, exist_ok=True)
                part = sub[sub["variable"] == v] if len(sub) else _EMPTY
                # An empty file is the cached fact "no data" (e.g. station
                # closed). It is only written after a SUCCESSFUL fetch: a
                # network error raises above and caches nothing.
                part.to_csv(p, index=False, compression="gzip")

    for sid in ids:
        for v in variables:
            df = _read_cache(_cached_file(cache_dir, provider.name, v, start, end, sid))
            if len(df):
                # a covering window holds more than was asked for
                frames.append(df[(df["time"] >= start) & (df["time"] <= end)])
    return pd.concat(frames, ignore_index=True) if frames else _EMPTY.copy()


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def load_stations(
    provider: StationProvider,
    region,
    start,
    end,
    variables: Sequence[str] = ("t2m",),
    cache_dir: str | Path = "outputs/cache/stations",
    qc: QCConfig | None = None,
    holdout_frac: float = 0.2,
    active: str = "full",
    include_unknown_start: bool = False,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Observations for every usable station in `region` over [start, end].

    active:
      "full"    -- station must have started by `start` and not ended before
                   `end` (default; makes completeness meaningful)
      "overlap" -- any overlap with the window
    include_unknown_start: stations whose start date is missing are skipped
      by default, because their history cannot be checked. For Meteo-France
      hourly this looks like the right default: of 683 such stations in the
      France box, a random sample of 12 returned NO data for 20-22 Aug 2023
      (their names end in -SPC, which suggests a separate flood-forecasting
      network). A sample is not proof, so the switch stays available; turning
      it on only costs download time, since completeness QC drops them.

    Returns (dataset, report). The report has one row per candidate station,
    with `kept` and a `reason`, so nothing disappears silently."""
    qc = qc or QCConfig()
    region = parse_region(region)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("start/end must be naive UTC timestamps")
    variables = list(variables)
    bad = [v for v in variables if v not in VARIABLES]
    if bad:
        raise KeyError(f"Unknown variable(s) {bad}; known: {sorted(VARIABLES)}")
    if qc.primary not in variables:
        raise ValueError(f"QC primary variable {qc.primary!r} must be in variables")
    if active not in ("full", "overlap"):
        raise ValueError("active must be 'full' or 'overlap'")

    # 1. candidates -------------------------------------------------------
    stations = provider.list_stations(region, start, end, variables).copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.drop_duplicates("station_id").set_index("station_id")
    n_listed = len(stations)

    s0, s1 = stations["start_date"], stations["end_date"]
    unknown = s0.isna()
    if active == "full":
        ok = (s0 <= start) & (s1.isna() | (s1 >= end))
    else:
        ok = (s0 <= end) & (s1.isna() | (s1 >= start))
    if include_unknown_start:
        ok = ok | (unknown & (s1.isna() | (s1 >= end if active == "full" else s1 >= start)))

    reason = pd.Series("", index=stations.index, dtype=object)
    reason[~ok & unknown] = "unknown start date"
    reason[~ok & ~unknown] = "not active over the window"
    logger.info("%d station(s) listed in region; %d active over the window "
                "(%d with unknown start date %s)", n_listed, int(ok.sum()),
                int(unknown.sum()), "included" if include_unknown_start else "skipped")
    ids = list(stations.index[ok])

    # 2. values -----------------------------------------------------------
    long = _fetch_with_cache(provider, ids, start, end, variables, cache_dir) if ids else _EMPTY.copy()

    # 3. assemble on a complete hourly axis -------------------------------
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="1h")
    cube = {}
    for v in variables:
        sub = long[long["variable"] == v]
        wide = (sub.pivot_table(index="time", columns="station_id", values="value", aggfunc="first")
                if len(sub) else pd.DataFrame(index=pd.DatetimeIndex([], name="time")))
        wide = wide.reindex(index=hours, columns=ids)
        lo, hi = (qc.ranges or {}).get(v, VARIABLES[v]["range"])
        out_of_range = ((wide < lo) | (wide > hi)).sum()
        wide = wide.where((wide >= lo) & (wide <= hi))   # NaN comparisons stay NaN
        cube[v] = (wide, out_of_range)

    # 4. QC ---------------------------------------------------------------
    prim, prim_oor = cube[qc.primary]
    completeness = prim.notna().mean() if len(ids) else pd.Series(dtype=float)
    report = pd.DataFrame({"candidate": stations.index.isin(ids)}, index=stations.index)
    report["completeness"] = completeness.reindex(stations.index)
    report["n_out_of_range"] = prim_oor.reindex(stations.index)
    kept_mask = completeness >= qc.min_completeness if len(ids) else pd.Series(dtype=bool)
    report["kept"] = kept_mask.reindex(stations.index).fillna(False).astype(bool)
    low = report["candidate"] & ~report["kept"]
    reason[low] = "completeness < %.2f" % qc.min_completeness
    report["reason"] = reason
    report = report.join(stations[["name", "latitude", "longitude", "elevation"]])
    kept = list(report.index[report["kept"]])
    logger.info("QC kept %d/%d candidate station(s)", len(kept), len(ids))

    # 5. dataset ----------------------------------------------------------
    meta = stations.loc[kept]
    ds = xr.Dataset(
        {v: (("time", "station"), cube[v][0][kept].to_numpy(dtype="f4")) for v in variables},
        coords={
            "time": hours.to_numpy().astype("datetime64[ns]"),
            "station": np.array(kept, dtype=object),
            "latitude": ("station", meta["latitude"].to_numpy("f8")),
            "longitude": ("station", meta["longitude"].to_numpy("f8")),
            "elevation": ("station", meta["elevation"].to_numpy("f8")),
            "name": ("station", meta["name"].astype(str).to_numpy(dtype=object)),
        },
    )
    ds["holdout"] = ("station", holdout_mask(kept, holdout_frac))
    for v in variables:
        ds[v].attrs["units"] = VARIABLES[v]["unit"]
    ds.attrs.update({
        "provider": provider.name,
        "license": provider.license,
        "window": f"{start} .. {end} (UTC, hourly, inclusive)",
        "region": list(region),
        "longitude_convention": "signed -180..180 (forecast zarrs use 0..360)",
        "qc_min_completeness": qc.min_completeness,
        "holdout_frac": holdout_frac,
    })
    return ds, report


def at_times(ds: xr.Dataset, times) -> xr.Dataset:
    """Observations at the given valid times (e.g. forecast valid times).

    Times with no observation row give NaN rather than raising, so a
    forecast step beyond the window simply scores nothing."""
    return ds.reindex(time=pd.DatetimeIndex(times).to_numpy().astype("datetime64[ns]"))