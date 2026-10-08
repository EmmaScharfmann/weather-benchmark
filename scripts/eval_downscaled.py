"""
Steps 7-8: AIFS-ENS vs AIFS-ENS + downscaler, at the Météo-France stations.

    uv run python scripts/eval_downscaled.py \
        --forecasts outputs/france_aug2025/aifs-ens1.0_*.zarr \
        --downscaled outputs/downscaled_aug2025/aifs-ens1.0_*.zarr \
        --region france --cache outputs/cache/stations --out outputs/eval_aug2025

Runs in the BASE environment (no torch needed: step 6 already produced the
downscaled fields).

Compared, at every station and forecast hour:
    aifs             the raw 0.25 deg ensemble, read at the station (bilinear)
    aifs_downscaled  the downscaled ensemble: each AIFS member x `sample` maps,
                     read at the station from the 5.5 km grid
    aifs_on_grid     the raw field after interpolation to the 5.5 km grid, read
                     the same way as aifs_downscaled (should ~equal `aifs`: a
                     check that the two reading paths agree)

Scores: RMSE/bias of the ensemble mean, fair CRPS, spread-skill ratio, by lead
time, valid hour and station elevation, with block-bootstrap intervals over
initialisations; Brier score and hit/false-alarm rates for heat thresholds.
Outputs: tables (CSV), scorecard.png, heatwave_map.png, summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline import station_interp as si
from weather_pipeline import station_scoring as ss
from weather_pipeline.regions import parse_region
from weather_pipeline.stations import MeteoFranceHourly, load_stations

logger = logging.getLogger("eval")
METHODS = ["aifs", "aifs_downscaled", "aifs_on_grid"]
THRESHOLDS_C = [30.0, 35.0]


# ---------------------------------------------------------------------------
def at_stations(fc_path: Path, ds_path: Path, stations: xr.Dataset) -> xr.Dataset:
    fc = xr.open_zarr(fc_path)["t2m"]
    if "ensemble" not in fc.dims:
        fc = fc.expand_dims(ensemble=[0])
    fc = fc.transpose("time", "lead_time", "ensemble", "lat", "lon").load().rename({"time": "init_time"})
    raw = si.interpolator_for(fc, stations).apply(fc).drop_vars("ensemble")             # (init, lead, ens, station)

    d = xr.open_zarr(ds_path)
    if not np.array_equal(d["time"].values, fc["init_time"].values) or not np.array_equal(d["lead_time"].values, fc["lead_time"].values):
        raise ValueError(f"{ds_path.name}: init/lead times differ from {fc_path.name}")
    ci = si.curvilinear_for(d["lat"].values, d["lon"].values, stations, k=4)
    if not ci.inside.all():
        logger.warning("%d station(s) further than 15 km from any CERRA cell -> NaN", int((~ci.inside).sum()))
    dsc = ci.apply(d["t2m_ds"].load().rename({"time": "init_time"}))                        # (init, lead, ens, sample, station)
    dsc = dsc.stack(member=("ensemble", "sample")).transpose("init_time", "lead_time", "member", "station")
    # its own dimension name: sharing "ensemble" with the 10-member raw forecast
    # would make xarray pad the raw one with NaN members (the scorer accepts any
    # dimension named ensemble*)
    dsc = dsc.drop_vars(["member", "ensemble", "sample"]).rename({"member": "ensemble_ds"})
    grid = ci.apply(d["t2m_interp"].load().rename({"time": "init_time"})).drop_vars("ensemble")   # (init, lead, ens, station)

    out = xr.Dataset({"aifs": raw, "aifs_downscaled": dsc, "aifs_on_grid": grid})
    out = out.assign_coords(station=stations["station"].values)
    for c in ("latitude", "longitude", "elevation"):
        out = out.assign_coords({c: ("station", stations[c].values)})
    return out.assign_coords(valid_time=(("init_time", "lead_time"),
                                         out["init_time"].values[:, None] + out["lead_time"].values[None, :]))


# ---------------------------------------------------------------------------
class EventScores:
    """Brier score and 50%-probability hit / false-alarm counts per lead and threshold."""

    def __init__(self, thresholds_K):
        self.thr = thresholds_K
        self.acc = {}       # (method, thr, lead_h) -> [sum (p-o)^2, n, hits, misses, false alarms, obs events]

    def add(self, fc_st: xr.Dataset, obs: xr.DataArray, methods):
        lead_h = (fc_st["lead_time"].values / np.timedelta64(1, "h")).astype(int)
        o = obs.values                                                                   # (init, lead, station)
        for m in methods:
            f = fc_st[m].values                                                          # (init, lead, ens, station)
            for t in self.thr:
                p = (f >= t).mean(2)
                ob = o >= t
                for j, h in enumerate(lead_h):
                    ok = np.isfinite(o[:, j]) & np.isfinite(p[:, j])
                    pj, oj = p[:, j][ok], ob[:, j][ok]
                    a = self.acc.setdefault((m, t, int(h)), np.zeros(6))
                    a += [((pj - oj) ** 2).sum(), ok.sum(), ((pj >= .5) & oj).sum(), ((pj < .5) & oj).sum(),
                          ((pj >= .5) & ~oj).sum(), oj.sum()]

    def table(self) -> pd.DataFrame:
        rows = []
        for (m, t, h), (b, n, hit, miss, fa, ev) in self.acc.items():
            rows.append({"method": m, "threshold_C": round(t - 273.15, 1), "lead_h": h, "n": int(n),
                         "obs_events": int(ev), "brier": b / n if n else np.nan,
                         "hit_rate": hit / (hit + miss) if hit + miss else np.nan,
                         "false_alarm_ratio": fa / (hit + fa) if hit + fa else np.nan})
        return pd.DataFrame(rows).sort_values(["threshold_C", "method", "lead_h"])


# ---------------------------------------------------------------------------
def run(fc_paths, ds_paths, region, cache_dir, out_dir, station_window=None, n_boot=200, provider=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for f in sorted(fc_paths):
        match = [d for d in ds_paths if Path(d).name == Path(f).name]
        if not match:
            raise FileNotFoundError(f"no downscaled file named {Path(f).name} among --downscaled")
        pairs.append((Path(f), Path(match[0])))
    ref = xr.open_zarr(pairs[0][0])
    inits = pd.DatetimeIndex([xr.open_zarr(f).time.values[0] for f, _ in pairs])
    start = inits.min().floor("h"); end = (inits.max() + pd.Timedelta(ref.lead_time.values[-1])).floor("h")
    if station_window:
        start, end = (pd.Timestamp(x) for x in station_window)
    stations, _ = load_stations(provider or MeteoFranceHourly(), region, start, end, variables=["t2m"], cache_dir=cache_dir)
    logger.info("%d stations, %d initialisations, window %s .. %s", stations.sizes["station"], len(pairs), start, end)

    cases_all, cases_bin, events = [], [], EventScores([t + 273.15 for t in THRESHOLDS_C])
    peak = None
    for f, d in pairs:
        fc_st = at_stations(f, d, stations)
        obs = si.observations_at_valid_times(fc_st, stations, "t2m")
        if peak is None:                       # consistency check on the first init
            diff = (fc_st["aifs_on_grid"] - fc_st["aifs"]).values
            logger.info("reading-path check: |aifs_on_grid - aifs| mean %.2f K, 95th pct %.2f K (should be small)",
                        np.nanmean(np.abs(diff)), np.nanpercentile(np.abs(diff), 95))
        cases_all.append(ss.score_station_case(fc_st, obs, "t2m", methods=METHODS, bin_by=None))
        cases_bin.append(ss.score_station_case(fc_st, obs, "t2m", methods=METHODS, bin_by="default"))
        events.add(fc_st, obs, ["aifs", "aifs_downscaled"])
        hot = np.nanmean(obs.values, axis=2)                                            # (init, lead) station-mean obs
        j = np.unravel_index(np.nanargmax(hot), hot.shape)
        if peak is None or hot[j] > peak[0]:
            peak = (hot[j], f, d, int(j[1]))
        logger.info("scored %s", f.name)

    res, resb = ss.combine_cases(cases_all), ss.combine_cases(cases_bin)
    ci = {m: ss.bootstrap_ci(cases_all, m, n_boot=n_boot) for m in ("aifs", "aifs_downscaled")} if len(cases_all) >= 3 else None

    # -- tables -------------------------------------------------------------------
    by_lead = pd.concat({m: res[m].to_dataframe() for m in METHODS}, names=["method"]).reset_index()
    by_lead["lead_h"] = (pd.to_timedelta(by_lead.pop("lead_time")).dt.total_seconds() / 3600).astype(int)
    by_lead["valid_hour"] = (by_lead["lead_h"] + inits[0].hour) % 24 if inits.hour.nunique() == 1 else np.nan
    by_lead.to_csv(out / "by_lead.csv", index=False)
    binned = pd.concat({m: resb[m].to_dataframe() for m in METHODS}, names=["method"]).reset_index()
    binned.to_csv(out / "by_region_elevation.csv", index=False)
    ev = events.table(); ev.to_csv(out / "heat_events.csv", index=False)

    a, b = res["aifs"], res["aifs_downscaled"]
    summary = {"n_inits": len(pairs), "n_stations": int(stations.sizes["station"]),
               "members": {"aifs": int(xr.open_zarr(pairs[0][0]).sizes.get("ensemble", 1)),
                           "aifs_downscaled": int(xr.open_zarr(pairs[0][1]).sizes["ensemble"] * xr.open_zarr(pairs[0][1]).sizes["sample"])},
               "rmse_K": {"aifs": float(a.rmse.mean()), "aifs_downscaled": float(b.rmse.mean())},
               "crps_K": {"aifs": float(a.crps.mean()), "aifs_downscaled": float(b.crps.mean())},
               "bias_K": {"aifs": float(a.bias.mean()), "aifs_downscaled": float(b.bias.mean())},
               "ssr": {"aifs": float(a.ssr.mean()), "aifs_downscaled": float(b.ssr.mean())}}
    summary["rmse_change_pct"] = 100 * (summary["rmse_K"]["aifs_downscaled"] / summary["rmse_K"]["aifs"] - 1)
    summary["crps_change_pct"] = 100 * (summary["crps_K"]["aifs_downscaled"] / summary["crps_K"]["aifs"] - 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    show = [h for h in (6, 24, 48, 96, 240) if h in by_lead.lead_h.values]
    print("\nAIFS-ENS vs AIFS-ENS + downscaler, all stations, density-weighted (K)")
    for metric in ("rmse", "crps", "ssr"):
        t = by_lead.pivot(index="lead_h", columns="method", values=metric).loc[show, ["aifs", "aifs_downscaled"]]
        print(f"\n{metric.upper()}\n{t.round(3).to_string()}")
    fr = binned[binned.region == "france"].groupby(["method", "elevation"]).rmse.mean().unstack(0)[["aifs", "aifs_downscaled"]]
    print(f"\nRMSE by station elevation band (lower edge, m), mean over leads\n{fr.round(2).to_string()}")
    e35 = ev[ev.threshold_C == 35].pivot(index="lead_h", columns="method", values="brier")
    if not e35.empty:
        print(f"\nBrier score, afternoon >= 35 °C (lower is better)\n{e35.loc[[h for h in show if h in e35.index]].round(3).to_string()}")
    print(f"\nOverall: RMSE {summary['rmse_K']['aifs']:.2f} -> {summary['rmse_K']['aifs_downscaled']:.2f} K "
          f"({summary['rmse_change_pct']:+.0f}%), CRPS {summary['crps_K']['aifs']:.2f} -> {summary['crps_K']['aifs_downscaled']:.2f} K "
          f"({summary['crps_change_pct']:+.0f}%), SSR {summary['ssr']['aifs']:.2f} -> {summary['ssr']['aifs_downscaled']:.2f}")

    _scorecard(res, resb, ci, ev, stations, out / "scorecard.png")
    _heatwave_map(peak, stations, out / "heatwave_map.png")
    print(f"\nwrote {out}/")
    return summary


def _scorecard(res, resb, ci, ev, stations, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hrs = lambda d: d["lead_time"].values / np.timedelta64(1, "h")
    col = {"aifs": "tab:blue", "aifs_downscaled": "tab:red"}
    lab = {"aifs": "AIFS-ENS", "aifs_downscaled": "AIFS-ENS + downscaler"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for metric, a, title in (("rmse", ax[0, 0], "RMSE of ensemble mean vs stations (K)"), ("crps", ax[0, 1], "fair CRPS vs stations (K)")):
        for m in ("aifs", "aifs_downscaled"):
            a.plot(hrs(res[m]), res[m][metric].values, color=col[m], marker="o", ms=3, label=lab[m])
            if ci:
                lo, hi = ci[m]; a.fill_between(hrs(res[m]), lo[metric].values, hi[metric].values, color=col[m], alpha=.15)
        a.set_title(title); a.set_xlabel("lead time (h)"); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    edges = ss.ELEVATION_EDGES
    (la0, la1), (lo0, lo1) = ss.FRANCE_REGIONS["france"]
    inbox = ((stations.latitude >= la0) & (stations.latitude <= la1) & (stations.longitude >= lo0) & (stations.longitude <= lo1)).values
    counts = [int((inbox & (stations.elevation.values >= lo) & (stations.elevation.values < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    for m in ("aifs", "aifs_downscaled"):
        d = resb[m].rmse.sel(region="france").mean("lead_time")
        ax[1, 0].plot(range(len(d)), d.values, marker="o", color=col[m], label=lab[m])
    ax[1, 0].set_xticks(range(len(counts))); ax[1, 0].set_xticklabels([f"{int(lo)}-{int(hi)} m\nn={n}" for lo, hi, n in zip(edges[:-1], edges[1:], counts)], fontsize=7)
    ax[1, 0].set_title("RMSE by station elevation (mean over leads)"); ax[1, 0].grid(alpha=.3); ax[1, 0].legend(fontsize=8)
    e = ev[ev.threshold_C == 35]
    for m in ("aifs", "aifs_downscaled"):
        d = e[e.method == m].sort_values("lead_h")
        ax[1, 1].plot(d.lead_h, d.brier, color=col[m], marker="o", ms=3, label=lab[m])
    ax[1, 1].set_title("Brier score, station temperature >= 35 °C (lower = better)"); ax[1, 1].set_xlabel("lead time (h)")
    ax[1, 1].grid(alpha=.3); ax[1, 1].legend(fontsize=8)
    fig.savefig(path, dpi=130); plt.close(fig)


def _heatwave_map(peak, stations, path):
    """For the hottest observed forecast hour: AIFS interpolated vs downscaled mean, with the stations' observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, f, d, j = peak
    ds = xr.open_zarr(d)
    vt = pd.Timestamp(ds["valid_time"].isel(time=0, lead_time=j).values)
    interp = ds["t2m_interp"].isel(time=0, lead_time=j).mean("ensemble").values - 273.15
    fine = ds["t2m_ds"].isel(time=0, lead_time=j).mean(("ensemble", "sample")).values - 273.15
    obs = stations["t2m"].sel(time=vt, method="nearest").values - 273.15
    lo, hi = np.nanpercentile(np.concatenate([fine.ravel(), obs[np.isfinite(obs)]]), [2, 98])
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for a, (t, fld) in zip(ax, [("AIFS-ENS mean, interpolated", interp), ("AIFS-ENS + downscaler, mean", fine)]):
        m = a.pcolormesh(ds.lon, ds.lat, fld, cmap="RdYlBu_r", vmin=lo, vmax=hi, shading="auto")
        a.scatter(stations.longitude, stations.latitude, c=obs, cmap="RdYlBu_r", vmin=lo, vmax=hi, s=9, edgecolors="k", linewidths=.2)
        a.set_title(f"{t}\n(dots: station observations)"); a.set_aspect(1 / np.cos(np.deg2rad(46)))
        fig.colorbar(m, ax=a, label="°C")
    fig.suptitle(f"valid {vt:%Y-%m-%d %H} UTC, init {pd.Timestamp(ds.time.values[0]):%Y-%m-%d} (+{int(ds.lead_time.values[j] / np.timedelta64(1, 'h'))} h)")
    fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecasts", nargs="+", required=True)
    p.add_argument("--downscaled", nargs="+", required=True)
    p.add_argument("--region", default="france")
    p.add_argument("--cache", default="outputs/cache/stations")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--station-window", default=None, metavar="START,END")
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("wetterdienst").setLevel(logging.WARNING)
    run(a.forecasts, a.downscaled, parse_region(a.region), a.cache, a.out,
        station_window=a.station_window.split(",") if a.station_window else None, n_boot=a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())