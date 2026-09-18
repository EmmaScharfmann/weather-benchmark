"""
Plotting utilities: rotating-globe forecast animation and RMSE line plot.
`render_rotating_gif` is a direct generalization of the original snippet --
same visuals, but parameterized by variable/title/units/cmap/unit-conversion
so it works for any model's output, not just Atlas + t2m.
"""

from __future__ import annotations

import io
import logging

import cartopy.crs as ccrs
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

CENTER_LON = 0 

logger = logging.getLogger(__name__)

FRAMES_PER_STEP = 12
N_TURNS = 1
FIGSIZE = (7, 6)
DPI = 120
CMAP = "RdYlBu_r"


def _valid_time_label(valid_time) -> str:
    return pd.Timestamp(valid_time).strftime("%Y-%m-%d %H:%M UTC")


def _render_frame(data, label, lon, norm, title, unit_label, cmap):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes(
        [0.03, 0.08, 0.72, 0.84],
        projection=ccrs.Orthographic(central_longitude=lon, central_latitude=20),
    )
    ax.set_global()
    ax.coastlines()
    data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        norm=norm,
        cmap=cmap,
        add_colorbar=False,
        add_labels=False,
    )
    ax.set_title(f"{title}\nValid time: {label}", fontsize=10)
    cax = fig.add_axes([0.80, 0.15, 0.03, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=unit_label)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return iio.imread(buf)


def render_rotating_gif(
    data: xr.DataArray,
    out_path: str,
    title: str = "Forecast",
    unit_label: str = "",
    convert_fn=None,
    cmap: str = CMAP,
    frames_per_step: int = FRAMES_PER_STEP,
    n_turns: int = N_TURNS,
    display: bool = False,
    center: float | None = None,
) -> str:
    """`data` must have dims (valid_time, latitude, longitude), e.g.
    `forecast["t2m"]` from `postprocess.load_forecast`.

    Set `center` (e.g. `center=0`) to force that value to sit at the
    midpoint of the colormap (white, for a diverging cmap like RdYlBu_r) --
    otherwise the colormap midpoint just falls wherever the 2nd/98th
    percentile range happens to put it.

    Set `display=True` to also render the GIF inline in a Jupyter notebook
    (in addition to saving it to `out_path`)."""
    if convert_fn is not None:
        data = convert_fn(data)

    n_steps = data.sizes["valid_time"]
    total_frames = n_steps * frames_per_step
    vmin, vmax = (float(x) for x in np.nanpercentile(data.values, [2, 98]))

    if center is not None:
        half_range = max(abs(vmin - center), abs(vmax - center))
        norm = mpl.colors.TwoSlopeNorm(vmin=center - half_range, vcenter=center, vmax=center + half_range)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    logger.info("Rendering %d frames (%d forecast steps x %d/step)", total_frames, n_steps, frames_per_step)

    frames = []
    for i in range(n_steps):
        frame_data = data.isel(valid_time=i)
        label = _valid_time_label(data["valid_time"].values[i])
        logger.info("  step %d/%d (%s)", i + 1, n_steps, label)
        for j in range(frames_per_step):
            frame_idx = i * frames_per_step + j
            lon = (frame_idx * 360 * n_turns / total_frames) % 360
            frames.append(
                _render_frame(frame_data, label, lon, norm, title, unit_label, cmap)
            )
    logger.info("Writing GIF to %s", out_path)
    iio.imwrite(out_path, frames, duration=220, loop=0)
    logger.info("Saved %s (%d frames)", out_path, len(frames))

    if display:
        from IPython.display import Image, display as ipy_display

        ipy_display(Image(filename=out_path))

    return out_path


def plot_rmse(rmse: pd.Series, variable: str, model_name: str, out_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(rmse.index, rmse.values, marker="o")
    ax.set_xlabel("Lead time [h]")
    ax.set_ylabel(f"RMSE ({variable})")
    ax.set_title(f"{model_name.upper()} \u2014 RMSE vs ERA5, {variable}")
    ax.grid(alpha=0.3)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig

    import io as _io


def recenter(da, center_lon=CENTER_LON):
    """Roll longitudes so `center_lon` sits in the middle of the map.

    Data is stored 0-360. Mapping to a window centred on `center_lon` and
    re-sorting avoids the discontinuity that would otherwise cut straight
    through the Atlantic -- pcolormesh draws a seam wherever longitude
    jumps, so the coordinate has to be monotonic across the new centre.
    """
    lon = ((da["longitude"].values - center_lon + 180) % 360) - 180
    return da.assign_coords(longitude=lon).sortby("longitude")
 

def ensemble_gif(da, label, cmap, dim, model,init_time, out_name, vmin=None, vmax=None,
                 center=None, duration=500, display=True):
    """Animate ensemble mean and spread over lead time.

    Colour scales are computed ONCE across all lead times and held fixed,
    so a colour means the same thing in every frame -- otherwise the growth
    of spread with lead time is invisible, because each frame would rescale
    to fill its own range.
    """
    import matplotlib as mpl

    da = recenter(da)

    mean_all = da.mean(dim=dim)
    spread_all = da.std(dim=dim, ddof=1)

    if center is not None:
        lim = float(np.nanpercentile(np.abs(mean_all.values - center), 99))
        norm = mpl.colors.TwoSlopeNorm(vmin=center - lim, vcenter=center,
                                       vmax=center + lim)
        mean_kw = dict(norm=norm)
    else:
        mean_kw = dict(
            vmin=float(np.nanpercentile(mean_all.values, 1)) if vmin is None else vmin,
            vmax=float(np.nanpercentile(mean_all.values, 99)) if vmax is None else vmax,
        )
    sp_max = float(np.nanpercentile(spread_all.values, 99))

    frames = []
    for i in range(da.sizes["valid_time"]):
        sel = da.isel(valid_time=i)
        valid = pd.Timestamp(da["valid_time"].values[i]).strftime("%a %d %b %H:%M UTC")

        fig = plt.figure(figsize=(12, 4.2), dpi=100)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.70, bottom=0.04, wspace=0.10)
        ax0, ax1 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        im0 = ax0.pcolormesh(sel.longitude, sel.latitude, sel.mean(dim=dim).values,
                             cmap=cmap, shading="auto", **mean_kw)
        im1 = ax1.pcolormesh(sel.longitude, sel.latitude,
                             sel.std(dim=dim, ddof=1).values,
                             cmap="magma", vmin=0, vmax=sp_max, shading="auto")

        for ax, im, title, sub, clabel in [
            (ax0, im0, "Ensemble mean", f"average of {da.sizes[dim]} members", label),
            (ax1, im1, "Ensemble spread", "where the members disagree",
             f"standard deviation [{label.split('[')[-1].rstrip(']')}]"),
        ]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbbbbb")
            ax.set_title(title, fontsize=11, pad=14, loc="left")
            ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5,
                    color="#555555", va="bottom")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                              fraction=0.048, pad=0.03, aspect=45)
            cb.set_label(clabel, fontsize=8.5)
            cb.ax.tick_params(labelsize=7.5, length=2)

        fig.text(0.02, 0.945, f"{model}  ·  {label.split(' [')[0]}",
                 fontsize=14, weight="medium", va="top")
        fig.text(0.02, 0.875,
                 f"{da.sizes[dim]}-member ensemble  ·  initialised "
                 f"{pd.Timestamp(init_time).strftime('%d %b %Y %H:%M UTC')}  ·  ERA5",
                 fontsize=9, color="#555555", va="top")
    

        buf = io.BytesIO()
        fig.savefig(buf, format="png")   # no bbox_inches: constant frame size
        plt.close(fig); buf.seek(0)
        frames.append(iio.imread(buf))

    path = f"../weather-benchmark/outputs/{out_name}"
    iio.imwrite(path, frames, duration=duration, loop=0)
    if display:
        from IPython.display import Image, display as ipy_display

        ipy_display(Image(filename=path))
