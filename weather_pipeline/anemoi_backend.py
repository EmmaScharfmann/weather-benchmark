"""
AIFS backends driven directly by anemoi-inference, bypassing Earth2Studio.

Why this exists separately from backends.py: Earth2Studio's AIFS wrapper
hardcodes a checkpoint version (0.18.0 pins v1.1, with no version kwarg),
which makes running v1.0 / v2.0 / the ensemble checkpoints awkward or
impossible. anemoi-inference loads whatever checkpoint you point it at and
derives variable ordering, grid, and timestep from that checkpoint's own
metadata -- so one code path covers every AIFS version.

The tradeoff is that we now own the grid handling Earth2Studio was doing
for us. AIFS runs on a reduced Gaussian grid (N320 for v1.x): anemoi's
state["fields"] values are 1-D arrays over grid points, NOT (lat, lon)
arrays. So this module interpolates:

    ERA5 regular lat/lon  --> model grid points   (building input state)
    model grid points     --> ERA5 regular grid   (writing output zarr)

so that the forecast zarr matches the schema postprocess.load_forecast and
scoring.py expect, identical to what the Earth2Studio models produce.

IMPORTANT -- interpolation error: RMSE computed on AIFS output therefore
includes round-trip interpolation error that Atlas/Aurora RMSE does not.
For a fair benchmark you should either (a) regrid every model to a common
grid the same way, or (b) score AIFS on its native grid against ERA5
interpolated to that grid. Neither is done here; this module just gets
AIFS running in the existing pipeline shape. Treat cross-model RMSE
comparisons as provisional until that's resolved.

API VERSION: written against anemoi-inference 0.12.0, whose surface was
verified directly (not guessed):
  * Runner.run(*, input_states=..., lead_time=...)  -- note the PLURAL
    kwarg; SimpleRunner accepts a bare state for single-dataset checkpoints
  * latitudes / longitudes / variable_to_input_tensor_index /
    model_computed_variables live on Metadata, reached via
    Checkpoint.multi_dataset_metadata -- NOT on Checkpoint itself
  * timestep / multi_step_input ARE on Checkpoint
Earlier releases spelled several of these differently, so if you pin a
different anemoi-inference, re-check these first.

SCORING CAVEAT (matters for a benchmark): AIFS is fine-tuned on IFS
analyses, so initialising it from ERA5 -- as this pipeline does -- degrades
its forecasts relative to how ECMWF runs it operationally. ECMWF flag this
explicitly. Aurora/Atlas do not necessarily share that penalty, so AIFS
numbers here are not directly comparable to theirs without a caveat.
"""

from __future__ import annotations

import functools
import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .regions import Region, crop_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# anemoi variable id  ->  the Earth2Studio-style channel id our
# ArraylakeERA5Source already knows how to resolve (see datasource.py's
# SURFACE_ALIASES / PRESSURE_ALIASES). anemoi names surface fields with
# GRIB-style short names and pressure levels as "<param>_<level>".
#
# Anything the checkpoint asks for that isn't resolvable here raises with
# the offending name, same philosophy as datasource.py's alias tables --
# add an entry rather than silently dropping a field the model needs.
# ---------------------------------------------------------------------------
ANEMOI_SURFACE_TO_E2S: dict[str, str] = {
    "2t": "t2m",
    "2d": "d2m",
    "10u": "u10m",
    "10v": "v10m",
    "100u": "u100m",
    "100v": "v100m",
    "msl": "msl",
    "sp": "sp",
    "tcw": "tcwv",
    "tp": "tp",
    "cp": "cp",
    # skt is SKIN temperature, NOT sea-surface temperature. (An earlier
    # version of this table mapped skt -> sst, which would have fed the
    # model the wrong field over land without any error.)
    "skt": "skt",
    "sst": "sst",
    "sd": "sd",
    # Soil levels: AIFS takes soil temperature/moisture as inputs.
    **{f"stl{i}": f"stl{i}" for i in range(1, 5)},
    **{f"swvl{i}": f"swvl{i}" for i in range(1, 5)},
    # Static surface fields. anemoi's low-level SimpleRunner API does NOT
    # inject these from the checkpoint -- prepare_input_tensor fails with
    # "Missing variables in input fields: ['z', 'lsm', 'sdor', 'slor']" if
    # we don't supply them, so they are fetched from ERA5 like anything
    # else. (The high-level DefaultRunner gets them from its configured
    # input source instead, which is why the docs make it look automatic.)
    #
    # Bare "z" here is SURFACE geopotential (orography x g), a different
    # field from pressure-level z_500 etc -- hence the distinct id
    # "z_surface", so datasource.py resolves it against the single-level
    # group rather than the pressure-level one.
    # NOTE: z / lsm / sdor / slor are NOT listed here. They're constant
    # fields, absent from the Arraylake repo, and come from the cached
    # statics file instead -- see weather_pipeline/statics.py.
}

# "<anemoi param>_<level>" -> Earth2Studio "<param><level>", e.g. t_500 -> t500
ANEMOI_PL_PARAMS: dict[str, str] = {
    "z": "z", "q": "q", "t": "t", "u": "u", "v": "v", "w": "w",
}

# Fields we must NOT try to fetch from ERA5, because anemoi supplies them
# itself: static fields baked into the checkpoint (surface geopotential,
# orography stats, land-sea mask) and computed forcings (time encodings,
# solar angle). Used in two places: when deciding what to request from the
# data source, and when collecting output fields (so these don't get
# written into the forecast zarr as if they were predictions).
#
# NOTE: _required_input_variables already drops meta.model_computed_variables,
# which covers most of the computed forcings on a well-formed checkpoint.
# This set is the belt-and-braces layer for checkpoints whose metadata
# doesn't declare them, plus the static fields, which are NOT in
# model_computed_variables but still must not be fetched.
ANEMOI_SKIP_FIELDS: set[str] = {
    "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
    "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
    "insolation", "cos_solar_zenith_angle",
    # NB: z / lsm / sdor / slor are deliberately NOT skipped. They looked
    # like checkpoint-supplied statics, but anemoi requires them in the
    # input state (see ANEMOI_SURFACE_TO_E2S above).
}
def anemoi_name_to_e2s(name: str) -> str:
    """Map one anemoi field id to the channel id ArraylakeERA5Source uses."""
    if name in ANEMOI_SURFACE_TO_E2S:
        return ANEMOI_SURFACE_TO_E2S[name]
    if "_" in name:
        param, _, level = name.rpartition("_")
        if param in ANEMOI_PL_PARAMS and level.isdigit():
            return f"{ANEMOI_PL_PARAMS[param]}{int(level)}"
    raise KeyError(
        f"Don't know how to map anemoi field '{name}' onto an ERA5 channel "
        "id -- add it to ANEMOI_SURFACE_TO_E2S / ANEMOI_PL_PARAMS in "
        "anemoi_backend.py, or to ANEMOI_SKIP_FIELDS if anemoi supplies it "
        "itself (static field or computed forcing)."
    )


def dataset_metadata(runner):
    """Get the object carrying latitudes/longitudes/variable indices.

    Two anemoi-inference API generations are in play, because the AIFS
    checkpoints span a wide range of training-time versions and each model
    env pins what its own checkpoint needs:

      * 0.12.x (used for 1.1 / 2.0): these live on a Metadata object,
        reached via Checkpoint.multi_dataset_metadata, a
        {dataset_name: Metadata} dict. Checkpoint itself does NOT expose
        them (verified by introspection).
      * 0.4.x (used for 1.0, whose pickle needs anemoi-models 0.3.x): the
        Checkpoint object exposes them directly.

    Returns whichever object answers, so callers just read attributes."""
    multi = getattr(runner.checkpoint, "multi_dataset_metadata", None)
    if multi:
        if len(multi) != 1:
            raise NotImplementedError(
                f"Checkpoint has {len(multi)} datasets ({list(multi)}); this "
                "backend only handles single-dataset checkpoints. Multi-dataset "
                "(e.g. cutout/LAM) models need per-dataset input states."
            )
        name, meta = next(iter(multi.items()))
        logger.info("Checkpoint dataset '%s' (anemoi >=0.12 layout)", name)
        return meta

    logger.info("Using Checkpoint directly (older anemoi layout)")
    return runner.checkpoint


def _required_input_variables(meta) -> list[str]:
    """Variables we must supply in the input state.

    Everything the input tensor indexes, minus what anemoi computes itself
    (solar/time forcings etc). VERIFIED: variable_to_input_tensor_index is a
    property on Metadata; model_computed_variables is a cached_property."""
    in_tensor = getattr(meta, "variable_to_input_tensor_index", None)
    if in_tensor is None:
        raise RuntimeError(
            "Could not read the input-variable index off this checkpoint. "
            f"Available: {sorted(a for a in dir(meta) if not a.startswith('_'))}"
        )
    in_tensor = list(in_tensor)
    computed = set(getattr(meta, "model_computed_variables", None) or ())
    wanted = [v for v in in_tensor if v not in computed]
    logger.info(
        "Input tensor wants %d variable(s); %d computed by anemoi; %d to fetch",
        len(in_tensor), len(computed), len(wanted),
    )
    return wanted


# ---------------------------------------------------------------------------
# Regridding, 0.25 deg lat/lon <-> the model's native grid (N320 for AIFS
# v1.x). Uses earthkit-regrid's sparse interpolation weights -- ECMWF's own
# matrices for exactly this pair of grids, downloaded once and cached on
# disk by earthkit-regrid itself.
#
# This is deliberately NOT a nearest-neighbour KDTree: N320 is a reduced
# Gaussian grid whose row lengths vary with latitude, and the proper
# weights are both more accurate and the supported path.
#
# Runs on CPU (scipy sparse matmul), so a GPU is only needed for the model
# forward pass itself.
# ---------------------------------------------------------------------------
SOURCE_GRID = (("grid", (0.25, 0.25)),)
TARGET_GRID = (("grid", "N320"),)

OUT_LAT = 90 - 0.25 * np.arange(721)
OUT_LON = 0.25 * np.arange(1440)


@functools.lru_cache(maxsize=4)
def _weights(source_grid, target_grid, method: str = "linear"):
    import earthkit.regrid as ekr
    from scipy.sparse import csr_matrix

    weights_csr, target_shape = ekr.db.find(dict(source_grid), dict(target_grid), method)
    matrix = csr_matrix(
        (weights_csr.data, weights_csr.indices, weights_csr.indptr),
        shape=weights_csr.shape,
    )
    return matrix, target_shape


def _regrid(data: np.ndarray, source_grid, target_grid) -> np.ndarray:
    matrix, target_shape = _weights(source_grid, target_grid)
    out = matrix @ np.asarray(data, dtype="f8").reshape(-1)
    return out.astype("f4").reshape(target_shape)


def to_n320(data) -> np.ndarray:
    """Regular 0.25 deg (lat, lon) -> 1-D array over the model's grid."""
    return _regrid(np.asarray(data), SOURCE_GRID, TARGET_GRID)


def to_latlon(data) -> np.ndarray:
    """1-D model-grid values -> regular 0.25 deg (lat, lon)."""
    return _regrid(np.asarray(data), TARGET_GRID, SOURCE_GRID)


@functools.lru_cache(maxsize=8)
def _crop_weights(region: Region):
    """Regridding matrix restricted to the rows inside `region`.

    The full model-grid -> 0.25 deg matrix has 721*1440 rows. Slicing it
    down to the region's rows BEFORE the matrix-vector product means a
    France-sized crop costs ~0.4% of a global regrid, instead of regridding
    the whole globe and throwing 99.6% of it away."""
    matrix, target_shape = _weights(TARGET_GRID, SOURCE_GRID)
    if tuple(target_shape) != (OUT_LAT.size, OUT_LON.size):
        raise RuntimeError(
            f"Regridding target shape {tuple(target_shape)} does not match the "
            f"assumed {(OUT_LAT.size, OUT_LON.size)} output grid; cropping "
            "indexes into that grid, so it cannot be applied."
        )
    lat_idx, lon_idx = crop_indices(region, OUT_LAT, OUT_LON)
    rows = (lat_idx[:, None] * OUT_LON.size + lon_idx[None, :]).ravel()
    return matrix[rows], (lat_idx.size, lon_idx.size), lat_idx, lon_idx


def to_latlon_crop(data, region: Region) -> np.ndarray:
    """1-D model-grid values -> the 0.25 deg (lat, lon) box `region` only."""
    matrix, shape, _, _ = _crop_weights(tuple(region))
    out = matrix @ np.asarray(data, dtype="f8").reshape(-1)
    return out.astype("f4").reshape(shape)


def crop_coords(region: Region) -> tuple[np.ndarray, np.ndarray]:
    """Latitude / longitude coordinate values of the cropped box."""
    _, _, lat_idx, lon_idx = _crop_weights(tuple(region))
    return OUT_LAT[lat_idx], OUT_LON[lon_idx]


@dataclass
class AnemoiBackend:
    """Mirrors backends.ModelBackend's `.run()` signature exactly, so
    pipeline.py and run_benchmark.py need no special-casing for AIFS."""

    name: str
    hf_repo: str
    n_members: int = 1
    # Output grid for the written zarr. 0.25 deg matches ERA5, so scoring
    # against fetch_truth needs no further regridding.
    out_lat: np.ndarray = field(default_factory=lambda: OUT_LAT)
    out_lon: np.ndarray = field(default_factory=lambda: OUT_LON)
    # Which output fields to keep, as ERA5/Earth2Studio channel ids ("t2m",
    # "u10m", ...) -- anemoi's own names ("2t") are accepted too. None keeps
    # every field the checkpoint produces (the old behaviour, which for a
    # ~100-field checkpoint is ~0.4 GB per step per member at global 0.25 deg).
    variables: list[str] | None = None
    # Crop box (lat_min, lat_max, lon_min, lon_max), lon in -180..180. None
    # keeps the whole globe. Applied inside the regridding matrix product, so
    # the discarded part is never computed, let alone stored.
    region: Region | None = None

    def _build_runner(self, device: str | None):
        import torch
        from anemoi.inference.runners.simple import SimpleRunner

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading anemoi checkpoint %s onto device=%s", self.hf_repo, device)
        # SimpleRunner accepts a HF repo spec dict in recent releases; if
        # your installed version doesn't, download explicitly instead:
        #   from huggingface_hub import hf_hub_download
        #   path = hf_hub_download(self.hf_repo, filename="inference-<...>.ckpt")
        #   SimpleRunner(path, device=device)
        return SimpleRunner({"huggingface": self.hf_repo}, device=device)

    def _build_input_state(self, runner, data_source, init_time: str) -> dict:
        """Fetch the ERA5 fields this checkpoint needs and put them on the
        model's grid.

        AIFS is a multi-step-input model: it needs the analysis at the init
        time AND the step before it (6h earlier for AIFS v1.x), so each
        field array is shaped (n_input_steps, n_grid_points)."""
        ckpt = runner.checkpoint
        meta = dataset_metadata(runner)
        # timestep / multi_step_input ARE on Checkpoint (verified 0.12.0)
        timestep = pd.Timedelta(ckpt.timestep)
        n_input_steps = int(ckpt.multi_step_input)
        init = pd.Timestamp(init_time)
        times = [init - i * timestep for i in reversed(range(n_input_steps))]
        logger.info(
            "Building input state: %d input step(s) %s, timestep=%s",
            n_input_steps, [str(t) for t in times], timestep,
        )

        from .statics import STATIC_FIELDS, load_statics

        wanted = _required_input_variables(meta)
        needed = [v for v in wanted if v not in ANEMOI_SKIP_FIELDS]
        # Split: constant fields come from the cached statics file, the rest
        # from the live ERA5 source.
        static_needed = [v for v in needed if v in STATIC_FIELDS]
        needed = [v for v in needed if v not in STATIC_FIELDS]

        # Resolve ALL names before failing, so one run tells you every field
        # that needs a mapping rather than one per run.
        e2s_ids, unmapped = [], []
        for v in needed:
            try:
                e2s_ids.append(anemoi_name_to_e2s(v))
            except KeyError:
                unmapped.append(v)
        if unmapped:
            raise KeyError(
                f"{len(unmapped)} anemoi field(s) have no ERA5 channel "
                f"mapping: {sorted(unmapped)}. Add each to "
                "ANEMOI_SURFACE_TO_E2S / ANEMOI_PL_PARAMS in "
                "anemoi_backend.py (and a matching entry in datasource.py's "
                "SURFACE_ALIASES so the id resolves against the Arraylake "
                "repo), or to ANEMOI_SKIP_FIELDS if anemoi supplies it "
                "itself. Full list this checkpoint asks for: "
                f"{sorted(needed)}"
            )
        logger.info("Checkpoint needs %d fetchable field(s)", len(needed))

        # One call per input time; data_source returns dims
        # (time, variable, lat, lon) -- same source used for truth/scoring.
        da = data_source(times, e2s_ids)

        fields: dict[str, np.ndarray] = {}

        if static_needed:
            # Read off the same open ERA5 source -- these are coordinates on
            # its single-level group, not a separate dataset.
            statics = load_statics(data_source)
            logger.info("Adding %d constant field(s): %s", len(static_needed), static_needed)
            for anemoi_name in static_needed:
                da_static = statics[anemoi_name]
                on_grid = to_n320(da_static.values)
                # constant in time: same values repeated for each input step
                fields[anemoi_name] = np.stack([on_grid] * len(times), axis=0)

        for anemoi_name, e2s_id in zip(needed, e2s_ids):
            per_step = [
                to_n320(da.sel(variable=e2s_id, drop=True).isel(time=i).values)
                for i in range(len(times))
            ]
            fields[anemoi_name] = np.stack(per_step, axis=0)

        return {"date": init.to_pydatetime(), "fields": fields}

    # ------------------------------------------------------------------
    # output selection
    # ------------------------------------------------------------------
    def _wants(self, anemoi_name: str) -> bool:
        """Is this output field one the caller asked to keep?"""
        if self.variables is None:
            return True
        wanted = set(self.variables)
        if anemoi_name in wanted:
            return True
        try:
            return anemoi_name_to_e2s(anemoi_name) in wanted
        except KeyError:
            return False

    def _check_variables_available(self, field_names) -> None:
        """Fail on the FIRST state, not after the whole forecast: a typo in
        --variables should cost seconds, not a 10-day rollout."""
        if self.variables is None:
            return
        have = {}
        for n in field_names:
            if n in ANEMOI_SKIP_FIELDS:
                continue
            have[n] = None
            try:
                have[anemoi_name_to_e2s(n)] = None
            except KeyError:
                pass
        missing = [v for v in self.variables if v not in have]
        if missing:
            raise ValueError(
                f"Requested variable(s) {missing} are not output fields of "
                f"{self.hf_repo}. Output fields: "
                f"{sorted(n for n in field_names if n not in ANEMOI_SKIP_FIELDS)}"
            )

    def _grid(self, values: np.ndarray) -> np.ndarray:
        """Model-grid values -> the output grid (cropped if a region is set)."""
        if self.region is None:
            return to_latlon(values)
        return to_latlon_crop(values, self.region)

    def _out_coords(self) -> tuple[np.ndarray, np.ndarray]:
        if self.region is None:
            return np.asarray(self.out_lat), np.asarray(self.out_lon)
        if not (np.array_equal(self.out_lat, OUT_LAT) and np.array_equal(self.out_lon, OUT_LON)):
            raise NotImplementedError(
                "Cropping assumes the default 0.25 deg output grid; a custom "
                "out_lat/out_lon cannot be combined with region."
            )
        return crop_coords(self.region)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        data_source,
        init_time: str,
        n_steps: int,
        out_zarr: str,
        device: str | None = None,
    ) -> str:
        runner = self._build_runner(device)
        input_state = self._build_input_state(runner, data_source, init_time)

        timestep = pd.Timedelta(runner.checkpoint.timestep)
        lead_time = n_steps * timestep
        lat, lon = self._out_coords()

        logger.info(
            "Running %s: init=%s, %d step(s) -> lead_time=%s, members=%d, "
            "variables=%s, region=%s",
            self.name, init_time, n_steps, lead_time, self.n_members,
            self.variables or "ALL", self.region or "global",
        )

        # Ensemble members come from the checkpoint's OWN stochastic
        # sampling: re-running with an identical input state produces a
        # different forecast each time. Measured on aifs-ens1.0, two runs
        # from the same ICs differed by 11.8 K max / 0.48 K mean on 2t --
        # far above GPU non-determinism (~1e-6), so this is real sampling.
        #
        # That means a genuine ensemble needs no perturbed initial
        # conditions: the input state is built once, outside this loop, and
        # each pass draws a new member. NOTE this means initial-condition
        # uncertainty is NOT sampled, so expect under-dispersion at short
        # lead times. It only holds for ens checkpoints; a deterministic
        # checkpoint re-run would give identical members, so n_members > 1
        # is meaningful only for aifs-ens*.
        #
        # Each member is written to the zarr as soon as it finishes, so peak
        # memory is one member's SELECTED fields, not every member's
        # every field.
        for member in range(self.n_members):
            t0 = time.perf_counter()
            collected: dict[str, list[np.ndarray]] = {}
            member_leads: list[pd.Timedelta] = []

            # anemoi >=0.12 takes input_states (plural, keyword-only);
            # 0.4.x took input_state. Try the modern spelling, fall back.
            try:
                stream = runner.run(input_states=input_state, lead_time=lead_time)
                stream_iter = iter(stream)
                first = next(stream_iter)
                stream = itertools.chain([first], stream_iter)
            except TypeError:
                stream = runner.run(input_state=input_state, lead_time=lead_time)

            for raw in stream:
                # SimpleRunner accepts a bare State for single-dataset
                # checkpoints (legacy path) and then yields bare States;
                # with an explicit dict it yields {dataset: State}.
                state = raw if "fields" in raw else next(iter(raw.values()))
                if member == 0 and not member_leads:
                    self._check_variables_available(state["fields"])
                valid = pd.Timestamp(state["date"])
                member_leads.append(valid - pd.Timestamp(init_time))
                for name, values in state["fields"].items():
                    if name in ANEMOI_SKIP_FIELDS or not self._wants(name):
                        continue
                    arr = np.asarray(values)
                    if arr.ndim > 1:      # some releases yield (1, n_points)
                        arr = arr.reshape(-1)
                    collected.setdefault(name, []).append(self._grid(arr))

            ds = self._member_dataset(
                collected, member_leads, init_time, member, lat, lon,
            )
            self._write_member(ds, out_zarr, member)
            logger.info(
                "%s: member %d/%d done in %.1fs (%d field(s), %d lead step(s))",
                self.name, member + 1, self.n_members,
                time.perf_counter() - t0, len(collected), len(member_leads),
            )

        # Sidecar marker: a crash mid-ensemble leaves a valid-looking zarr
        # with fewer members than requested; only this file says it finished.
        Path(str(out_zarr).rstrip("/") + ".done").write_text(
            f"{self.name} members={self.n_members}\n"
        )
        logger.info("%s run complete. Forecast written to %s", self.name, out_zarr)
        return out_zarr

    def _write_member(self, ds: xr.Dataset, out_zarr: str, member: int) -> None:
        if member == 0:
            ds.to_zarr(out_zarr, mode="w")
        else:
            ds.to_zarr(out_zarr, mode="a", append_dim="ensemble")

    def _member_dataset(
        self,
        collected: dict[str, list[np.ndarray]],
        lead_times: list[pd.Timedelta],
        init_time: str,
        member: int,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> xr.Dataset:
        """One member in the schema Earth2Studio's ZarrBackend writes, so
        postprocess.load_forecast and wbx_scoring.open_forecast handle it
        unchanged: dims (time, lead_time, lat, lon), one data variable per
        channel id. With more than one member the dataset carries a length-1
        `ensemble` dim, so successive members append along it to give
        (time, lead_time, ensemble, lat, lon)."""
        init = np.datetime64(pd.Timestamp(init_time))
        lead = np.array([pd.Timedelta(t).to_timedelta64() for t in lead_times])

        data_vars = {}
        for anemoi_name in sorted(collected):
            # store under the Earth2Studio-style id, so downstream code and
            # the other models' zarrs use identical variable names
            try:
                out_name = anemoi_name_to_e2s(anemoi_name)
            except KeyError:
                logger.warning(
                    "Output field '%s' has no ERA5 channel mapping; storing "
                    "under its anemoi name", anemoi_name,
                )
                out_name = anemoi_name
            arr = np.stack(collected[anemoi_name], axis=0)  # (lead, lat, lon)
            if self.n_members == 1:
                data_vars[out_name] = (("time", "lead_time", "lat", "lon"), arr[None])
            else:
                data_vars[out_name] = (
                    ("time", "lead_time", "ensemble", "lat", "lon"),
                    arr[None, :, None],
                )

        coords = {"time": np.array([init]), "lead_time": lead, "lat": lat, "lon": lon}
        if self.n_members > 1:
            coords["ensemble"] = np.array([member])
        ds = xr.Dataset(data_vars, coords=coords)
        # Attributes go on EVERY member's dataset, not just the first: each
        # append rewrites the group attributes, so a member written without
        # them would wipe the ones the first member set.
        ds.attrs.update({
            "model": self.name,
            "hf_repo": self.hf_repo,
            "init_time": str(pd.Timestamp(init_time)),
            "n_members": self.n_members,
            "region": "global" if self.region is None else list(self.region),
            "lon_convention": "0-360; non-monotonic if the region crosses 0",
            "ic_note": "members differ by model sampling only; the input "
                       "state is identical, so initial-condition "
                       "uncertainty is not sampled",
        })
        return ds


def aifs_backend(version: str, n_members: int = 1) -> AnemoiBackend:
    """version: "1.0" | "1.1" | "2.0" | "ens1.0"."""
    repos = {
        "1.0": "ecmwf/aifs-single-1.0",
        "1.1": "ecmwf/aifs-single-1.1",
        # 2.0 checkpoints are deliberately absent: they need WAVE fields
        # that ERA5 does not contain (and ENS v2 additionally requires w on
        # all pressure levels despite its card listing w as diagnostic), so
        # they cannot be initialised from this data source at all.
        # NOTE on ens2.0 / single 2.0: these expect WAVE fields that ERA5
        # does not contain, and ENS v2 additionally requires w on all
        # pressure levels despite its model card listing w as diagnostic.
        # Expect _required_input_variables to ask for fields
        # ArraylakeERA5Source cannot supply; that is a data problem, not a
        # code one.
        # VERIFIED to exist on huggingface.co
        "ens1.0": "ecmwf/aifs-ens-1.0",
    }
    if version not in repos:
        raise ValueError(f"Unknown AIFS version {version!r}; choose from {list(repos)}")
    name = f"aifs{version}" if not version.startswith("ens") else f"aifs-{version}"
    return AnemoiBackend(name=name, hf_repo=repos[version], n_members=n_members)