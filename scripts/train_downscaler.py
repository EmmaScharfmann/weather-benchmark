"""
Step 4: train the downscaler (ERA5 on the CERRA grid -> CERRA), validate
against plain interpolation.

    # full run, once data/pairs/pairs_france.zarr exists (train 2014-2019, validate 2020)
    uv run --with-requirements requirements/downscaler.txt \
        python scripts/train_downscaler.py --pairs data/pairs --out runs/crps

    # the same model trained with mean squared error, for the comparison
    ... --loss mse --out runs/mse

    # code check on the months built so far (before consolidation)
    ... --pairs data/pairs --out runs/dev --val-start 2014-05-01 --epochs 2

Outputs in --out:
    model.pt          weights + normalisation + fine-grid terrain (all inference needs)
    history.json      per-epoch training loss and validation scores
    val_report.json   final validation: model vs interpolation, overall, by hour, by elevation
    curves.png, val_maps.png

Validation scores are in K, against CERRA, over the validation period:
    rmse      error of the ensemble mean (interpolation: the field itself)
    crps      fair CRPS of the ensemble (interpolation, one member: equals its MAE)
    spread    ensemble standard deviation; ssr = spread / rmse (1 = well calibrated)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from weather_pipeline import downscaler as dsc

logger = logging.getLogger("train")
ELEV_EDGES = [0, 200, 500, 1000, 1500, 2500, 5000]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def open_pairs(pairs_dir: Path) -> xr.Dataset:
    full = pairs_dir / "pairs_france.zarr"
    if full.exists():
        return xr.open_zarr(full)
    months = sorted((pairs_dir / "months").glob("pairs_*.zarr"))
    if not months:
        raise FileNotFoundError(f"no pairs in {pairs_dir}")
    logger.warning("pairs not consolidated: using the %d month(s) built so far (fine for a code check)", len(months))
    return xr.open_mfdataset([str(m) for m in months], engine="zarr", combine="nested", concat_dim="time")


def split_times(times: pd.DatetimeIndex, stats_file: Path, val_start, val_end):
    if val_start:
        v0 = pd.Timestamp(val_start); v1 = pd.Timestamp(val_end) if val_end else times.max()
    elif stats_file.exists() and json.loads(stats_file.read_text()).get("val_years"):
        vy = json.loads(stats_file.read_text())["val_years"]
        v0, v1 = pd.Timestamp(f"{min(vy)}-01-01"), pd.Timestamp(f"{max(vy)}-12-31T23:59")
    else:
        v0, v1 = times[int(len(times) * 0.8)], times.max()
        logger.warning("no validation period given: using the last 20%% of times (%s ..)", v0)
    val = (times >= v0) & (times <= v1)
    train = times < v0
    if not train.any() or not val.any():
        raise ValueError(f"empty split: train {train.sum()}, val {val.sum()} (val {v0}..{v1})")
    return train, val


def get_stats(stats_file: Path, era5_tr, cerra_tr, static) -> dict:
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
    else:
        logger.warning("stats.json missing (not consolidated): computing from the training slice")
        r = cerra_tr - era5_tr
        dh = static["cerra_orography"] - static["era5_orography"]
        stats = {"era5_t2m": {"mean": float(era5_tr.mean()), "std": float(era5_tr.std())},
                 "residual": {"mean": float(r.mean()), "std": float(r.std())},
                 "cerra_orography": {"mean": float(static["cerra_orography"].mean()), "std": float(static["cerra_orography"].std())},
                 "orography_diff": {"mean": float(dh.mean()), "std": float(dh.std())}}
    for k in ("lat", "lon"):
        stats[k] = {"mean": float(static[k].mean()), "std": float(static[k].std())}
    return stats


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class Scores:
    """Accumulates squared error, CRPS and variance sums, overall and per group."""

    def __init__(self):
        self.s = {}

    def add(self, key, err2, crps, var, n):
        a = self.s.setdefault(key, np.zeros(4))
        a += [err2, crps, var, n]

    def table(self):
        out = {}
        for k, (e2, c, v, n) in self.s.items():
            rmse, spread = np.sqrt(e2 / n), np.sqrt(v / n)
            out[k] = {"rmse": round(float(rmse), 4), "crps": round(float(c / n), 4),
                      "spread": round(float(spread), 4), "ssr": round(float(spread / rmse), 3) if rmse > 0 else None,
                      "n": int(n)}
        return out


def _crps_np(ens, y):
    """fair CRPS per pixel, ens (B, M, H, W), y (B, H, W)."""
    m = ens.shape[1]
    skill = np.abs(ens - y[:, None]).mean(1)
    if m == 1:
        return skill
    spread = np.abs(ens[:, :, None] - ens[:, None]).sum((1, 2)) / (2 * m * (m - 1))
    return skill - spread


@torch.no_grad()
def validate(net, era5, cerra, times, static_norm, stats, members, device, groups=None, batch=8):
    net.eval()
    model, base = Scores(), Scores()
    hours = pd.DatetimeIndex(times).hour.values
    for i in range(0, len(era5), batch):
        c, y = era5[i:i + batch], cerra[i:i + batch]
        x = torch.from_numpy(dsc.make_inputs(c, times[i:i + batch], static_norm, stats)).to(device)
        ens = dsc.to_kelvin(c, net(x, members).cpu().numpy(), stats)               # (B, M, H, W) K
        mean = ens.mean(1)
        e2, cr = (mean - y) ** 2, _crps_np(ens, y)
        var = ens.var(1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(mean)
        b2, bc = (c - y) ** 2, np.abs(c - y)
        sel = {"all": np.ones(y.shape[1:], bool)}
        if groups is not None:
            sel.update(groups)
        for name, mask in sel.items():
            n = mask.sum() * len(y)
            model.add(name, e2[:, mask].sum(), cr[:, mask].sum(), var[:, mask].sum(), n)
            base.add(name, b2[:, mask].sum(), bc[:, mask].sum(), 0.0, n)
            if groups is not None and name == "land":
                for h in np.unique(hours[i:i + batch]):
                    k = hours[i:i + batch] == h
                    nh = mask.sum() * k.sum()
                    model.add(f"land_{h:02d}UTC", e2[k][:, mask].sum(), cr[k][:, mask].sum(), var[k][:, mask].sum(), nh)
                    base.add(f"land_{h:02d}UTC", b2[k][:, mask].sum(), bc[k][:, mask].sum(), 0.0, nh)
    net.train()
    return model.table(), base.table()


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="data/pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--loss", choices=["crps", "mse"], default="crps")
    p.add_argument("--members", type=int, default=4, help="ensemble members per input during training (crps)")
    p.add_argument("--eval-members", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--noise-channels", type=int, default=4)
    p.add_argument("--val-start", default=None); p.add_argument("--val-end", default=None)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="cap for quick code checks")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = Path(a.pairs)

    # -- data (in memory: 7 years of 6-hourly France is ~3-4 GB) ----------------
    ds = open_pairs(pairs)
    st = xr.open_zarr(pairs / "static.zarr").load()
    static = {k: st[k].values.astype("f4") for k in ("cerra_orography", "era5_orography", "land_sea_mask")}
    static["lat"], static["lon"] = st.lat.values.astype("f4"), st.lon.values.astype("f4")
    times = pd.DatetimeIndex(ds.time.values)
    tr, va = split_times(times, pairs / "stats.json", a.val_start, a.val_end)
    t0 = time.time()
    era5, cerra = ds.era5_t2m.values.astype("f4"), ds.cerra_t2m.values.astype("f4")
    logger.info("loaded %d steps (%d train, %d val), grid %dx%d in %.0fs",
                len(times), tr.sum(), va.sum(), era5.shape[1], era5.shape[2], time.time() - t0)
    stats = get_stats(pairs / "stats.json", era5[tr], cerra[tr], static)
    static_norm = dsc.static_stack(static, stats)
    tr_idx = np.flatnonzero(tr)
    v_era5, v_cerra, v_times = era5[va], cerra[va], times[va].to_numpy()
    land = static["land_sea_mask"] > 0.5
    groups = {"land": land}
    for lo, hi in zip(ELEV_EDGES[:-1], ELEV_EDGES[1:]):
        groups[f"land_{lo}-{hi}m"] = land & (static["cerra_orography"] >= lo) & (static["cerra_orography"] < hi)
    groups = {k: v for k, v in groups.items() if v.any()}

    # -- model -----------------------------------------------------------------
    cfg = dsc.Config(loss=a.loss, base=a.base, depth=a.depth, noise_channels=a.noise_channels if a.loss == "crps" else 0)
    net = dsc.DownscalerNet(cfg).to(device)
    n_par = sum(p_.numel() for p_ in net.parameters())
    steps_per_epoch = len(tr_idx) // a.batch
    if a.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, a.max_steps_per_epoch)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, a.epochs * steps_per_epoch))
    loss_fn = dsc.fair_crps if cfg.stochastic else dsc.mse
    m_train = a.members if cfg.stochastic else 1
    m_eval = a.eval_members if cfg.stochastic else 1
    logger.info("model: %s loss, %.2fM parameters, %d steps/epoch, device %s", a.loss, n_par / 1e6, steps_per_epoch, device)

    # -- baseline (interpolation) once ---------------------------------------------
    _, base_val = validate(net, v_era5[:1], v_cerra[:1], v_times[:1], static_norm, stats, 1, device)   # warm-up
    best, history = None, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); order = rng.permutation(tr_idx); losses = []
        for s in range(steps_per_epoch):
            idx = np.sort(order[s * a.batch:(s + 1) * a.batch])
            c, y = era5[idx], cerra[idx]
            x = torch.from_numpy(dsc.make_inputs(c, times[idx].to_numpy(), static_norm, stats)).to(device)
            target = torch.from_numpy(dsc.residual_norm(c, y, stats)).to(device)
            loss = loss_fn(net(x, m_train), target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); losses.append(float(loss))
        mv, bv = validate(net, v_era5, v_cerra, v_times, static_norm, stats, m_eval, device)
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "val": mv["all"], "baseline": bv["all"],
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        logger.info("epoch %d/%d  loss %.4f  val rmse %.3f K (interp %.3f)  crps %.3f K (interp %.3f)  ssr %s  [%.0fs]",
                    ep, a.epochs, rec["train_loss"], mv["all"]["rmse"], bv["all"]["rmse"], mv["all"]["crps"],
                    bv["all"]["crps"], mv["all"]["ssr"], rec["seconds"])
        score = mv["all"]["crps"] if cfg.stochastic else mv["all"]["rmse"]
        if best is None or score < best:
            best = score
            dsc.Downscaler(net, stats, static, device).save(out / "model.pt", extra={"epoch": ep, "args": vars(a)})
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # -- final report with the best checkpoint ---------------------------------------
    best_model = dsc.Downscaler.load(out / "model.pt", device)
    mv, bv = validate(best_model.net, v_era5, v_cerra, v_times, static_norm, stats, m_eval, device, groups=groups)
    report = {"loss": a.loss, "val_period": [str(times[va].min()), str(times[va].max())],
              "best_epoch": torch.load(out / "model.pt", weights_only=False)["extra"]["epoch"],
              "model": mv, "interpolation": bv}
    (out / "val_report.json").write_text(json.dumps(report, indent=2))
    _figures(out, history, best_model, v_era5, v_cerra, v_times, m_eval)
    rows = [(k, bv[k]["rmse"], mv[k]["rmse"], mv[k]["crps"], mv[k]["ssr"]) for k in mv]
    print("\nvalidation (K)        interp rmse   model rmse   model crps   ssr")
    for k, b, r, c, s in rows:
        print(f"  {k:20s} {b:10.3f}  {r:11.3f}  {c:11.3f}   {s}")
    improved = mv["all"]["rmse"] < bv["all"]["rmse"]
    print(f"\n{'model beats interpolation' if improved else 'model does NOT beat interpolation yet'}; outputs in {out}/")
    return 0


def _figures(out, history, model, v_era5, v_cerra, v_times, members):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val"]["rmse"] for h in history], marker="o", label="model rmse")
    ax[1].plot(ep, [h["val"]["crps"] for h in history], marker="s", label="model crps")
    ax[1].axhline(history[0]["baseline"]["rmse"], color="k", ls="--", label="interpolation rmse")
    ax[1].axhline(history[0]["baseline"]["crps"], color="gray", ls=":", label="interpolation crps (= mae)")
    ax[1].set_title("validation vs CERRA (K)"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    fig.savefig(out / "curves.png", dpi=110); plt.close(fig)

    hours = pd.DatetimeIndex(v_times).hour
    mid = np.flatnonzero(hours == 12) if (hours == 12).any() else np.arange(len(v_times))
    i = int(mid[np.argmax(v_cerra[mid].mean((1, 2)))])                       # hottest validation midday
    ens = model.predict(v_era5[i:i + 1], v_times[i:i + 1], members=max(members, 1))[0]
    panels = [("interpolated ERA5", v_era5[i]), ("CERRA (truth)", v_cerra[i]), ("model, member 1", ens[0]),
              ("model, ensemble mean", ens.mean(0))]
    lo, hi = np.percentile(v_cerra[i], [1, 99]) - 273.15
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6), constrained_layout=True)
    for a, (t, f) in zip(ax, panels):
        m = a.pcolormesh(model.lon, model.lat, f - 273.15, cmap="RdYlBu_r", vmin=lo, vmax=hi, shading="auto")
        a.set_title(t); fig.colorbar(m, ax=a, label="°C")
    sd = ens.std(0, ddof=1) if ens.shape[0] > 1 else np.zeros_like(ens[0])
    m = ax[4].pcolormesh(model.lon, model.lat, sd, cmap="magma", shading="auto")
    ax[4].set_title("ensemble spread (std)"); fig.colorbar(m, ax=ax[4], label="K")
    for a in ax:
        a.set_aspect(1 / np.cos(np.deg2rad(46)))
    fig.suptitle(f"validation, {pd.Timestamp(v_times[i]):%Y-%m-%d %H UTC}")
    fig.savefig(out / "val_maps.png", dpi=95); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())