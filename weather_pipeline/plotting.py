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