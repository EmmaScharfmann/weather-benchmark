"""
Build a single self-contained HTML comparison report from whatever
run_benchmark.py has already scored.

Reads outputs/scores/*.csv and outputs/fields/*.nc -- no Arraylake access,
no forecast zarrs, no GPU. So the report is cheap to regenerate and can run
anywhere, including CI.

## Aggregation across init times

Scores are read from the per-case ACCUMULATOR files, averaged across init
times, and finalised exactly once at the end. That ordering is the whole
point: RMSE does not average, so mean(RMSE over 10 dates) is a different
and wrong number. Metrics that declared themselves non-accumulable
(activity_ratio) have no accumulator file column; those are averaged from
their finalised per-case values and flagged in the report, because for a
ratio that is an approximation rather than the exact multi-case value.

Images are embedded as base64 PNGs so the HTML is one portable file.

Usage:
    uv run scripts/make_report.py --variable t2m
    uv run scripts/make_report.py --variable t2m --baseline aifs1.1
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from weather_pipeline.scoring import REGISTRY, finalize_accumulators

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = REPO_ROOT / "outputs/scores"
FIELDS_DIR = REPO_ROOT / "outputs/fields"

# <model>_<variable>_<initslug>_accumulators.csv -- model names contain dots
# and hyphens (aifs1.0, aifs-ens1.0), so match the fixed tail and take the
# rest as the model rather than splitting on "_".
_ACC_RE = re.compile(r"^(?P<model>.+)_(?P<variable>[^_]+)_(?P<init>\d{8}T\d{4})_accumulators\.csv$")
_FIN_RE = re.compile(r"^(?P<model>.+)_(?P<variable>[^_]+)_(?P<init>\d{8}T\d{4})_metrics\.csv$")


def _collect(pattern: re.Pattern, variable: str) -> dict[str, dict[str, pd.DataFrame]]:
    """{model: {init_slug: DataFrame}} for one variable."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    if not SCORES_DIR.exists():
        return out
    for path in sorted(SCORES_DIR.glob("*.csv")):
        m = pattern.match(path.name)
        if not m or m.group("variable") != variable:
            continue
        df = pd.read_csv(path, index_col="lead_time_h")
        out.setdefault(m.group("model"), {})[m.group("init")] = df
    return out


def aggregate(variable: str) -> tuple[dict[str, pd.DataFrame], dict[str, int], list[str]]:
    """Per-model aggregated scores, case counts, and which columns were
    approximated (non-accumulable metrics averaged after finalisation)."""
    accs = _collect(_ACC_RE, variable)
    fins = _collect(_FIN_RE, variable)

    aggregated: dict[str, pd.DataFrame] = {}
    n_cases: dict[str, int] = {}
    approximated: set[str] = set()

    for model in sorted(set(accs) | set(fins)):
        per_case_acc = list(accs.get(model, {}).values())
        per_case_fin = list(fins.get(model, {}).values())
        n_cases[model] = len(per_case_fin or per_case_acc)

        frames = []
        if per_case_acc:
            # correct path: average inner values, THEN finalise
            mean_acc = sum(per_case_acc) / len(per_case_acc)
            frames.append(finalize_accumulators(mean_acc))
        if per_case_fin:
            extra = [c for c in per_case_fin[0].columns
                     if not per_case_acc or c not in per_case_acc[0].columns]
            if extra:
                approximated.update(extra)
                frames.append(sum(f[extra] for f in per_case_fin) / len(per_case_fin))
        if frames:
            aggregated[model] = pd.concat(frames, axis=1)

    return aggregated, n_cases, sorted(approximated)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_skill_lines(aggregated, metric: str, variable: str) -> str | None:
    """Metric vs lead time, one line per model. The workhorse figure: where
    a model crosses a baseline is visible at a glance."""
    series = {m: df[metric] for m, df in aggregated.items() if metric in df}
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, s in sorted(series.items()):
        ax.plot(s.index, s.values, marker="o", ms=4, label=model)
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(f"{metric} ({variable})")
    if metric == "spread_skill_ratio":
        ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_title(f"{metric} vs lead time")
    ax.grid(alpha=0.3)
    if metric == "bias":
        ax.axhline(0, color="k", lw=0.8, ls="--")
    if metric == "activity_ratio":
        ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.legend(fontsize=8)
    return _png(fig)


def plot_scorecard(aggregated, metric: str, baseline: str | None) -> str | None:
    """Models x lead time heatmap.

    With a baseline, cells are % improvement over it (diverging, centred at
    zero) -- which is the ECMWF-style scorecard and the figure that scales
    as models and lead times are added. Without one, raw values."""
    series = {m: df[metric] for m, df in aggregated.items() if metric in df}
    if len(series) < 1:
        return None

    frame = pd.DataFrame(series).T.sort_index()
    if baseline and baseline in frame.index:
        base = frame.loc[baseline]
        # lower-is-better for every metric here except activity_ratio, which
        # has no "better" direction -- so it gets no baseline treatment.
        if metric == "activity_ratio":
            data, cmap, center, label = frame, "viridis", None, metric
        else:
            data = 100 * (base - frame) / base
            data = data.drop(index=baseline)
            cmap, center = "RdBu_r", 0
            label = f"% improvement in {metric} vs {baseline}"
    else:
        data, cmap, center, label = frame, "viridis", None, metric

    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(data.columns) + 3), 0.6 * len(data) + 2.2))
    vmax = np.nanmax(np.abs(data.values)) if center == 0 else None
    im = ax.imshow(data.values, aspect="auto", cmap=cmap,
                   vmin=-vmax if center == 0 else None,
                   vmax=vmax if center == 0 else None)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_xlabel("Lead time [h]")
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = data.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_title(label)
    return _png(fig)


def plot_bias_maps(variable: str) -> str | None:
    """Time-mean error map per model, shared diverging colour scale.

    Shows what the scalars cannot: whether error is concentrated over land,
    in the tropics, or at the poles. Averaged over the cached per-case
    fields, so adding init times sharpens rather than multiplies panels."""
    if not FIELDS_DIR.exists():
        return None
    per_model: dict[str, list[xr.DataArray]] = {}
    for path in sorted(FIELDS_DIR.glob(f"*_{variable}_*_bias.zarr")):
        name = path.name[: -len("_bias.zarr")]
        model = name.rsplit(f"_{variable}_", 1)[0]
        ds = xr.open_zarr(path)
        per_model.setdefault(model, []).append(ds[list(ds.data_vars)[0]].load())
    if not per_model:
        return None

    means = {m: sum(v) / len(v) for m, v in sorted(per_model.items())}
    vmax = max(float(np.nanpercentile(np.abs(d.values), 98)) for d in means.values())

    n = len(means)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.1 * nrows), squeeze=False)
    for ax, (model, da) in zip(axes.ravel(), means.items()):
        im = ax.pcolormesh(da["longitude"], da["latitude"], da.values,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"{model} — mean bias", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, label=f"{variable} bias (forecast - truth)", shrink=0.8)
    return _png(fig)


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
        padding: 0 1rem; color: #222; line-height: 1.5; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.2rem; margin-top: 2.2rem; }}
 table {{ border-collapse: collapse; font-size: 0.86rem; }}
 th, td {{ border: 1px solid #ddd; padding: 3px 8px; text-align: right; }}
 th {{ background: #f4f4f4; }} td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; margin: 0.6rem 0; }}
 .note {{ background: #fffbe6; border-left: 3px solid #e0c060; padding: 0.6rem 0.9rem;
          font-size: 0.88rem; }}
 .meta {{ color: #666; font-size: 0.85rem; }}
</style></head><body>
<h1>{title}</h1>
<p class="meta">Variable <b>{variable}</b> &middot; generated {generated} &middot; {ncases}</p>
{notes}
{sections}
</body></html>
"""


def build_html(variable: str, baseline: str | None, title: str) -> str:
    aggregated, n_cases, approximated = aggregate(variable)
    if not aggregated:
        raise SystemExit(
            f"No scores found for variable '{variable}' in {SCORES_DIR}. "
            "Run scripts/run_benchmark.py first."
        )

    # Spread-skill ratio: derived from two aggregated metrics rather than
    # stored per case, because the ratio of averages is what you want, not
    # the average of ratios. sqrt((K+1)/K) is the standard finite-ensemble
    # correction; without the member count we can only report the raw
    # ratio, so note that in the report rather than silently omitting it.
    for model, df in aggregated.items():
        if "spread" in df and "rmse" in df:
            df["spread_skill_ratio"] = df["spread"] / df["rmse"]

    metrics = sorted({c for df in aggregated.values() for c in df.columns},
                     key=lambda m: list(REGISTRY).index(m) if m in REGISTRY else 99)

    notes = []
    counts = ", ".join(f"{m}: {n}" for m, n in sorted(n_cases.items()))
    if len(set(n_cases.values())) > 1:
        notes.append(
            "<div class='note'><b>Unequal case counts.</b> Models were scored "
            f"over different numbers of init times ({counts}). Cross-model "
            "differences partly reflect which dates each model covers, not "
            "only skill.</div>"
        )
    if approximated:
        notes.append(
            "<div class='note'><b>Approximated across cases:</b> "
            f"{', '.join(approximated)}. These metrics are not linearly "
            "averageable, so their multi-case values are the mean of "
            "per-case values rather than an exact aggregate.</div>"
        )
    if any("spread_skill_ratio" in df for df in aggregated.values()):
        notes.append(
            "<div class='note'><b>Spread-skill ratio</b> is spread / "
            "ensemble-mean RMSE, computed from the aggregated values. A "
            "calibrated ensemble sits near 1; below 1 is under-dispersed "
            "(overconfident), the usual AI-ensemble failure. This is the raw "
            "ratio -- published figures often apply a sqrt((K+1)/K) "
            "finite-ensemble correction, which raises it slightly for small "
            "member counts.</div>"
        )
    notes.append(
        "<div class='note'><b>Comparability.</b> AIFS models are initialised "
        "from ERA5 though ECMWF fine-tuned them on IFS analyses, and their "
        "output is regridded N320&rarr;0.25&deg;. Both penalties are absent "
        "for Aurora and Atlas, so cross-family gaps overstate the skill "
        "difference.</div>"
    )

    sections = []

    table = pd.concat({m: df for m, df in sorted(aggregated.items())}, axis=1)
    sections.append("<h2>Aggregated scores</h2>" + table.round(4).to_html())

    sections.append("<h2>Skill vs lead time</h2>")
    for metric in metrics:
        png = plot_skill_lines(aggregated, metric, variable)
        if png:
            sections.append(f'<img alt="{metric} vs lead time" src="data:image/png;base64,{png}">')

    sections.append("<h2>Scorecard</h2>")
    if baseline:
        sections.append(f"<p class='meta'>Positive = better than <b>{baseline}</b>.</p>")
    for metric in metrics:
        png = plot_scorecard(aggregated, metric, baseline)
        if png:
            sections.append(f'<img alt="{metric} scorecard" src="data:image/png;base64,{png}">')

    maps = plot_bias_maps(variable)
    if maps:
        sections.append("<h2>Mean bias maps</h2>")
        sections.append(
            "<p class='meta'>Time-mean forecast minus truth, averaged over all "
            "scored init times. Shared colour scale.</p>"
        )
        sections.append(f'<img alt="bias maps" src="data:image/png;base64,{maps}">')

    return TEMPLATE.format(
        title=title, variable=variable,
        generated=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        ncases=f"init times per model — {counts}",
        notes="\n".join(notes), sections="\n".join(sections),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", default="t2m")
    parser.add_argument(
        "--baseline", default=None,
        help="Model to express the scorecard relative to (e.g. a climatology "
             "or persistence baseline once those exist).",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    html = build_html(
        args.variable, args.baseline,
        args.title or f"Weather model benchmark — {args.variable}",
    )
    out = Path(args.out or REPO_ROOT / f"outputs/report_{args.variable}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Wrote %s (%.0f kB)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()