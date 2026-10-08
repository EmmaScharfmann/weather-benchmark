"""
M0 driver: forecast zarrs -> station baselines -> scores -> checks, tables, figure.

Run on ONE zarr and a small region as the smoke test (a couple of minutes):

    uv run python scripts/m0_scorecard.py \
        --zarr outputs/smoke.zarr --region 45.0,46.5,4.0,6.0 \
        --cache outputs/cache/stations_smoke --out outputs/m0_smoke

and on every init, over all of France, for the real thing:

    uv run python scripts/m0_scorecard.py \
        --zarr outputs/france_aug2023/aifs-ens1.0_*.zarr --region france \
        --cache outputs/cache/stations --out outputs/m0

Runs in the BASE environment (weatherbenchX, wetterdienst, arraylake) -- the
forecasts were already produced by the per-model environments.

Output files go in --out: metrics_overall.csv, metrics_binned.csv,
scorecard.png and checks.txt. The exit code is 1 if any structural check
FAILs. WARN lines are things that look odd scientifically; they are
judgement calls, not errors -- read them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline import station_interp as si
from weather_pipeline import station_scoring as ss
from weather_pipeline.regions import parse_region
from weather_pipeline.stations import MeteoFranceHourly, QCConfig, load_stations

logger = logging.getLogger("m0")

LADDER = ["nearest", "bilinear", "bilinear_lapse", "analysis_bilinear", "analysis_bilinear_lapse"]


class Checks:
    def __init__(self):
        self.lines, self.failed = [], False

    def __call__(self, level: str, name: str, ok: bool, detail: str = ""):
        tag = "PASS" if ok else level            # level: FAIL or WARN
        self.failed |= (not ok and level == "FAIL")
        line = f"[{tag}] {name}" + (f" -- {detail}" if detail else "")
        self.lines.append(line)
        print(line)
        return ok


def _open_forecasts(paths, variable, check):
    fcs = []
    for p in paths:
        ds = xr.open_zarr(p)
        check("FAIL", f"{Path(p).name}: has {variable} with dims time/lead_time/lat/lon",
              variable in ds and {"time", "lead_time", "lat", "lon"} <= set(ds[variable].dims),
              f"dims={dict(ds[variable].sizes) if variable in ds else 'variable missing'}")
        check("WARN", f"{Path(p).name}: finished run (.done marker present)",
              Path(str(p).rstrip("/") + ".done").exists(), "a crash mid-ensemble leaves a partial zarr")
        fcs.append(ds)
    ref = fcs[0]
    for p, ds in zip(paths[1:], fcs[1:]):
        check("FAIL", f"{Path(p).name}: same grid and lead times as the first zarr",
              np.array_equal(ds.lat, ref.lat) and np.array_equal(ds.lon, ref.lon)
              and np.array_equal(ds.lead_time, ref.lead_time))
    return fcs


def run(zarr_paths, region, cache_dir, out_dir, variable="t2m", n_boot=200, block=2,
        provider=None, era5_source=None, orography=None, min_completeness=0.9, make_plot=True,
        station_window=None) -> bool:
    """Returns True if no structural check failed. provider / era5_source /
    orography are injectable so the glue can be tested without a network."""
    t_start = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    check = Checks()
    paths = [str(p) for p in zarr_paths]
    fcs = _open_forecasts(paths, variable, check)
    ref = fcs[0]

    # -- value sanity (K) and ensemble ----------------------------------------
    v0 = ref[variable].isel(time=0, lead_time=0)
    lo, hi = float(v0.min()), float(v0.max())
    check("FAIL", "forecast values are in Kelvin (230-330 K)", 230 < lo and hi < 330, f"{lo:.1f}..{hi:.1f}")
    n_ens = ref.sizes.get("ensemble", 1)
    print(f"      {len(paths)} init(s), {n_ens} member(s), lead steps {ref.sizes['lead_time']}, "
          f"first lead {pd.Timedelta(ref.lead_time.values[0])}, grid {ref.sizes['lat']}x{ref.sizes['lon']}")

    inits = pd.DatetimeIndex(np.concatenate([d.time.values for d in fcs]))
    start = inits.min().floor("h")
    end = (inits.max() + pd.Timedelta(ref.lead_time.values[-1])).floor("h")
    if station_window:
        start, end = (pd.Timestamp(x) for x in station_window)
    print(f"      station window: {start} .. {end}")

    # -- stations ---------------------------------------------------------------
    provider = provider or MeteoFranceHourly()
    stations, report = load_stations(provider, region, start, end, variables=[variable],
                                     cache_dir=cache_dir, qc=QCConfig(min_completeness=min_completeness))
    n_st = stations.sizes["station"]
    check("FAIL", "enough stations loaded (>= 5)", n_st >= 5, f"{n_st} kept of {int(report.candidate.sum())} candidates")
    check("WARN", "most candidate stations pass QC", n_st >= 0.7 * max(int(report.candidate.sum()), 1),
          f"{n_st}/{int(report.candidate.sum())}")
    report.to_csv(out / "station_qc_report.csv")

    # -- orography ---------------------------------------------------------------
    if era5_source is None:
        from weather_pipeline.datasource import build_data_source
        era5_source = build_data_source("era5")
    if orography is None:
        from weather_pipeline.statics import load_statics
        orography = si.height_from_geopotential(load_statics(era5_source)["z"])
    oro = si.orography_on_grid(orography, ref)
    check("FAIL", "model orography looks like metres (-500..6000)", -500 < float(oro.min()) and float(oro.max()) < 6000,
          f"{float(oro.min()):.0f}..{float(oro.max()):.0f} m -- if ~1e4+, geopotential was not divided by g")

    # -- per-init baselines and scores -------------------------------------------
    cases_all, cases_bin, first = [], [], True
    for p, ds in zip(paths, fcs):
        t0 = time.time()
        fc_st = si.downscale_to_stations(ds, stations, variable, orography=oro)
        obs = si.observations_at_valid_times(fc_st, stations, variable)
        an = si.analysis_baseline(era5_source, ds, stations, variable, orography=oro)
        if first:
            first = False
            n_out = int((~np.isfinite(fc_st.bilinear.isel(init_time=0, lead_time=0).values.reshape(-1, n_st).mean(0))).sum())
            check("WARN", "all stations inside the forecast grid", n_out == 0, f"{n_out} outside")
            hd = np.abs(fc_st.h_diff.values)
            check("WARN", "station-vs-model height difference is sane (95th pct < 1500 m)",
                  np.nanpercentile(hd, 95) < 1500, f"median {np.nanmedian(hd):.0f} m, 95th pct {np.nanpercentile(hd, 95):.0f} m")
            frac = float(np.isfinite(obs.values).mean())
            check("FAIL", "observations found for forecast valid times", frac > 0, f"{frac:.0%} finite")
            check("WARN", "observation coverage >= 80%", frac >= 0.8, f"{frac:.0%}")
            if n_ens > 1:
                sp = fc_st.bilinear.std("ensemble", ddof=1).mean(("init_time", "station"))
                check("WARN", "ensemble spread grows with lead time", float(sp[-1]) > float(sp[0]),
                      f"{float(sp[0]):.2f} K at first lead -> {float(sp[-1]):.2f} K at last")
        cases_all.append(ss.score_station_case(fc_st, obs, variable, analysis=an, bin_by=None))
        cases_bin.append(ss.score_station_case(fc_st, obs, variable, analysis=an, bin_by="default"))
        print(f"      scored {Path(p).name} in {time.time() - t0:.0f}s")

    res = ss.combine_cases(cases_all)          # all stations, per lead time
    resb = ss.combine_cases(cases_bin)         # region x elevation bins

    # -- science-level warnings (judgement, not facts) ----------------------------
    f6 = {m: float(res[m].rmse.isel(lead_time=0)) for m in res}
    check("WARN", "first-lead RMSE of the analysis floor is plausible (< 5 K)",
          f6.get("analysis_bilinear_lapse", 0) < 5, f"{f6.get('analysis_bilinear_lapse', float('nan')):.2f} K")
    if "bilinear_lapse" in f6 and "bilinear" in f6:
        better = float(res["bilinear_lapse"].rmse.mean()) < float(res["bilinear"].rmse.mean())
        check("WARN", "lapse correction lowers mean RMSE vs plain bilinear (it may legitimately not, "
              "e.g. inversions at night)", better,
              f"{float(res['bilinear'].rmse.mean()):.2f} -> {float(res['bilinear_lapse'].rmse.mean()):.2f} K")

    # -- tables ----------------------------------------------------------------------
    long = []
    for m, ds in res.items():
        d = ds.to_dataframe().reset_index()
        d.insert(0, "method", m); d["lead_h"] = (pd.to_timedelta(d.pop("lead_time")).dt.total_seconds() / 3600).astype(int)
        long.append(d)
    pd.concat(long).to_csv(out / "metrics_overall.csv", index=False)
    pd.concat([ds.to_dataframe().reset_index().assign(method=m) for m, ds in resb.items()]).to_csv(
        out / "metrics_binned.csv", index=False)

    show = [h for h in (6, 24, 48, 96, 240) if h in ss.to_frame(res, "rmse").index]
    pd.set_option("display.width", 140)
    for metric in ("rmse", "bias"):
        print(f"\n{metric.upper()} (K), all stations, density-weighted, at lead hours {show}")
        print(ss.to_frame(res, metric).loc[show].round(2).to_string())
    ens_methods = {m: ds for m, ds in res.items() if "ssr" in ds}
    if ens_methods:
        print("\nSPREAD-SKILL RATIO (1 = calibrated; <1 = under-dispersed) and fair CRPS (K)")
        print(pd.concat({"ssr": ss.to_frame(ens_methods, "ssr").loc[show],
                         "crps": ss.to_frame(ens_methods, "crps").loc[show]}, axis=1).round(2).to_string())

    ci = None
    if len(cases_all) >= 3 and n_boot:
        ci = {m: ss.bootstrap_ci(cases_all, m, n_boot=n_boot, block=block) for m in ("bilinear", "bilinear_lapse") if m in res}
        print(f"\n(bootstrap CIs over {len(cases_all)} inits computed; illustrative -- one event, few inits)")

    if make_plot:
        _plot(res, resb, ci, inits, stations, out / "scorecard.png")
    (out / "checks.txt").write_text("\n".join(check.lines) + "\n")
    print(f"\nwrote {out}/  ({time.time() - t_start:.0f}s total); "
          f"{'SOME CHECKS FAILED' if check.failed else 'no structural failures'}")
    return not check.failed


def _plot(res, resb, ci, inits, stations, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    hrs = lambda ds: ds["lead_time"].values / np.timedelta64(1, "h")
    style = {"nearest": ("tab:gray", "--"), "bilinear": ("tab:blue", "-"), "bilinear_lapse": ("tab:green", "-"),
             "analysis_bilinear": ("tab:orange", ":"), "analysis_bilinear_lapse": ("tab:red", ":")}
    for metric, a, title in (("rmse", ax[0, 0], "RMSE vs stations (K)"), ("bias", ax[0, 1], "Bias, forecast - obs (K)")):
        for m in LADDER:
            if m in res:
                c, ls = style[m]; a.plot(hrs(res[m]), res[m][metric].values, color=c, ls=ls, marker="o", ms=3, label=m)
        if ci and metric in ("rmse", "bias") and "bilinear_lapse" in ci:
            lo, hi = ci["bilinear_lapse"]
            a.fill_between(hrs(res["bilinear_lapse"]), lo[metric].values, hi[metric].values, color="tab:green", alpha=.2)
        a.set_title(title); a.set_xlabel("lead time (h)"); a.grid(alpha=.3)
        if metric == "bias": a.axhline(0, color="k", lw=.8)
    hours = {int(pd.Timestamp(t).hour) for t in inits}
    if len(hours) == 1:
        # one tick per lead step, labelled with the VALID hour (UTC): the diurnal
        # pattern in the errors is the thing to read off these panels
        h0 = next(iter(hours)); ticks = hrs(res["bilinear"])
        for a in (ax[0, 0], ax[0, 1]):
            a.set_xticks(ticks); a.set_xticklabels([f"{(h0 + int(t)) % 24:02d}" for t in ticks], fontsize=6)
            a.set_xlabel("valid hour (UTC); lead time increases to the right")
    ax[0, 0].legend(fontsize=8)

    edges = ss.ELEVATION_EDGES
    (la0, la1), (lo0, lo1) = ss.FRANCE_REGIONS["france"]
    in_box = ((stations.latitude >= la0) & (stations.latitude <= la1) & (stations.longitude >= lo0) & (stations.longitude <= lo1)).values
    counts = [int((in_box & (stations.elevation.values >= lo) & (stations.elevation.values < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    labels = [f"{int(lo)}-{int(hi)} m\nn={n}" for lo, hi, n in zip(edges[:-1], edges[1:], counts)]
    xs = {int(lo): i for i, lo in enumerate(edges[:-1])}
    for m in ("bilinear", "bilinear_lapse"):
        if m in resb:
            d = resb[m].rmse.sel(region="france").mean("lead_time")
            ax[1, 0].plot([xs[int(e)] for e in d["elevation"].values], d.values, marker="o", color=style[m][0], label=m)
    ax[1, 0].set_xticks(range(len(labels))); ax[1, 0].set_xticklabels(labels, fontsize=7)
    ax[1, 0].set_title("RMSE by elevation band (n = stations)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    if "ssr" in res.get("bilinear_lapse", {}):
        r = res["bilinear_lapse"]; a = ax[1, 1]
        l1, = a.plot(hrs(r), r["ssr"].values, color="tab:purple", marker="o", ms=3, label="spread-skill ratio (left)")
        a.axhline(1, color="k", lw=.8, ls="--"); a.set_ylabel("SSR  (1 = calibrated, <1 = under-dispersed)")
        a.set_ylim(0, max(1.1, float(np.nanmax(r["ssr"].values)) * 1.1)); a.set_xlabel("lead time (h)")
        b = a.twinx(); l2, = b.plot(hrs(r), r["crps"].values, color="tab:brown", marker="s", ms=3, label="fair CRPS, K (right)")
        b.set_ylabel("CRPS (K)"); b.set_ylim(bottom=0)
        a.set_title("Ensemble calibration and CRPS (bilinear_lapse)"); a.grid(alpha=.3)
        a.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    else:
        ax[1, 1].axis("off")
    fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", nargs="+", required=True)
    p.add_argument("--region", default="france", help="name or lat_min,lat_max,lon_min,lon_max (lon -180..180)")
    p.add_argument("--cache", default="outputs/cache/stations")
    p.add_argument("--out", default="outputs/m0")
    p.add_argument("--variable", default="t2m")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--min-completeness", type=float, default=0.9)
    p.add_argument("--station-window", default=None, metavar="START,END",
                   help="override the observation window (default: first init .. last init + max lead)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("wetterdienst").setLevel(logging.WARNING)
    ok = run(sorted(a.zarr), parse_region(a.region), a.cache, a.out, a.variable, a.n_boot,
             min_completeness=a.min_completeness,
             station_window=a.station_window.split(",") if a.station_window else None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())