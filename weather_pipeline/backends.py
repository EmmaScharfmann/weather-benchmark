"""
Per-model backends. Each backend just knows which Earth2Studio model class
to load -- the data source (ArraylakeERA5Source) and the run/IO logic are
shared, since every prognostic model in Earth2Studio implements the same
`input_coords()` / `__call__` interface.

Model-specific extras (static fields, native grids, checkpoint variants) are
handled internally by each model's own `load_model()` -- we don't need to
know about them here.

Imports for each model are done lazily inside the factory functions so you
only need the optional dependency for whichever model you're actually
running (e.g. `pip install earth2studio[aurora]`).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Callable

import torch

from .datasource import ArraylakeERA5Source

# NOTE: earth2studio is deliberately NOT imported at module level. The AIFS
# backends run in envs that contain anemoi and no earth2studio at all, and
# pipeline.py imports this module for get_backend() whichever model is
# selected -- a top-level import here would break every AIFS run. Imported
# inside ModelBackend.run() instead, which only earth2studio models reach.

logger = logging.getLogger(__name__)


@dataclass
class ModelBackend:
    name: str
    load_default_package: Callable
    load_model: Callable

    def load(self, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s package/checkpoint...", self.name)
        package = self.load_default_package()
        model = self.load_model(package)
        logger.info("Moving %s onto device=%s", self.name, device)
        return model.to(device)

    def run(
        self,
        data_source: ArraylakeERA5Source,
        init_time: str,
        n_steps: int,
        out_zarr: str,
        device: str | None = None,
    ) -> str:
        from earth2studio.io import ZarrBackend
        from earth2studio.run import deterministic as run_deterministic

        model = self.load(device)
        io = ZarrBackend(out_zarr)
        logger.info(
            "Running %s: init_time=%s, n_steps=%d -> %s", self.name, init_time, n_steps, out_zarr
        )
        run_deterministic([init_time], n_steps, model, data_source, io)
        logger.info("%s run complete. Forecast written to %s", self.name, out_zarr)
        return out_zarr


def _aurora_backend() -> ModelBackend:
    from earth2studio.models.px import Aurora

    return ModelBackend("aurora", Aurora.load_default_package, Aurora.load_model)


@dataclass
class StochasticModelBackend(ModelBackend):
    """An Earth2Studio model whose forward pass is stochastic, run N times
    to build an ensemble.

    Aurora 1.5's ensemble checkpoint injects fresh Gaussian noise into the
    backbone conditioning on every forward pass, so repeated calls from the
    SAME initial condition give statistically independent members -- the
    same mechanism as the AIFS ensemble checkpoint, and it likewise needs
    no perturbed ICs.

    The model is loaded ONCE and reused across members: it is a ~48 GB
    checkpoint, so reloading per member would dominate runtime.

    Members are run into separate zarr stores and then combined, because
    Earth2Studio's `deterministic` runner owns its output store and has no
    member concept. The intermediates are removed afterwards."""

    n_members: int = 1

    def run(
        self,
        data_source: ArraylakeERA5Source,
        init_time: str,
        n_steps: int,
        out_zarr: str,
        device: str | None = None,
    ) -> str:
        import shutil
        from pathlib import Path

        import xarray as xr
        from earth2studio.io import ZarrBackend
        from earth2studio.run import deterministic as run_deterministic

        if self.n_members == 1:
            return super().run(data_source, init_time, n_steps, out_zarr, device)

        model = self.load(device)
        tmp_dir = Path(out_zarr).parent / f".{Path(out_zarr).name}.members"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        member_paths = []
        try:
            for i in range(self.n_members):
                logger.info("%s: member %d/%d", self.name, i + 1, self.n_members)
                path = tmp_dir / f"member_{i:03d}.zarr"
                run_deterministic([init_time], n_steps, model, data_source,
                                  ZarrBackend(str(path)))
                member_paths.append(path)

            logger.info("Combining %d members into %s", len(member_paths), out_zarr)
            members = [xr.open_zarr(p) for p in member_paths]
            combined = xr.concat(members, dim="ensemble")
            combined = combined.assign_coords(ensemble=range(len(members)))
            combined.to_zarr(out_zarr, mode="w")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("%s run complete. Forecast written to %s", self.name, out_zarr)
        return out_zarr


def _aurora1p5_backend(ensemble: bool = False, n_members: int = 1):
    """Aurora 1.5.

    NOTE on comparability: Aurora 1.5 was pretrained on ERA5 and fine-tuned
    on IFS operational analyses -- the same lineage as AIFS. So unlike the
    original Aurora (ERA5 throughout), initialising v1.5 from ERA5 carries
    the same mismatch penalty the AIFS models do. That makes v1.5-vs-AIFS a
    fairer comparison than v1-vs-AIFS was, and it means the ERA5-vs-IFS
    truth question now applies to both families rather than only one."""
    if ensemble:
        from earth2studio.models.px import Aurora1p5Ensemble as Model
        name = "aurora1.5-ens"
    else:
        from earth2studio.models.px import Aurora1p5 as Model
        name = "aurora1.5"
    return StochasticModelBackend(
        name, Model.load_default_package, Model.load_model, n_members=n_members
    )


def _atlas_backend(crps: bool = False, n_members: int = 1):
    """Atlas, NVIDIA's generative weather model.

    Atlas is diffusion-like: it samples a low-resolution latent state
    through a STOCHASTIC INTERPOLANT, so even the plain checkpoint is not
    deterministic -- repeated calls from the same initial condition give
    different forecasts. That makes it usable as an ensemble the same way
    aifs-ens1.0 and aurora1.5-ens are.

    `crps=True` selects Atlas CRPS, the CRPS-TRAINED ensemble variant,
    where each forward pass draws a fresh noise vector modulating every
    transformer block. NVIDIA score it as a 16-member ensemble. Prefer this
    one for probabilistic work: a CRPS-trained model is calibrated by
    construction, whereas sampling the deterministic-intent checkpoint is
    not guaranteed to give meaningful spread."""
    if crps:
        try:
            from earth2studio.models.px import AtlasCRPS as Model
        except ImportError as exc:
            from earth2studio.models import px
            raise ImportError(
                "AtlasCRPS not found in this earth2studio. Available px "
                f"models: {sorted(n for n in dir(px) if not n.startswith('_'))}. "
                "It may be named differently or need a newer release."
            ) from exc
        name = "atlas-crps"
    else:
        from earth2studio.models.px import Atlas as Model
        name = "atlas"

    return StochasticModelBackend(
        name, Model.load_default_package, Model.load_model, n_members=n_members
    )


# AIFS no longer goes through Earth2Studio: its wrapper hardcodes a
# checkpoint version, which made running v1.0 / v2.0 / ensemble awkward.
# anemoi_backend drives anemoi-inference directly and derives everything
# (variables, grid, timestep) from whichever checkpoint is loaded, so one
# code path covers all AIFS versions. These factories return an
# AnemoiBackend, which exposes the same `.run()` signature as ModelBackend.
def _aifs_anemoi(version: str, n_members: int = 1):
    from .anemoi_backend import aifs_backend

    return aifs_backend(version, n_members=n_members)


BACKEND_FACTORIES: dict[str, Callable[[], ModelBackend]] = {
    "aurora": _aurora_backend,
    "aurora1.5": lambda: _aurora1p5_backend(ensemble=False),
    # Stochastic checkpoint: repeated forward passes from the same initial
    # condition give independent members, so --n-members works here the
    # same way it does for aifs-ens1.0.
    "aurora1.5-ens": lambda: _aurora1p5_backend(ensemble=True, n_members=1),
    # Generative (stochastic interpolant), so --n-members works here too.
    "atlas": lambda: _atlas_backend(n_members=1),
    # CRPS-trained ensemble variant -- the right Atlas for probabilistic
    # comparison against aifs-ens1.0 / aurora1.5-ens.
    "atlas-crps": lambda: _atlas_backend(crps=True, n_members=1),
    "aifs1.0": lambda: _aifs_anemoi("1.0"),
    "aifs1.1": lambda: _aifs_anemoi("1.1"),
    # Ensemble checkpoint. Default of 1 member keeps a bare run cheap;
    # pass --n-members to draw a real ensemble (the checkpoint samples
    # stochastically, so repeated runs differ -- see AnemoiBackend.run).
    "aifs-ens1.0": lambda: _aifs_anemoi("ens1.0", n_members=1),
}


def get_backend(name: str, n_members: int | None = None,
                variables: list[str] | None = None, region=None):
    """n_members only applies to backends that support it (the anemoi
    ensemble path). Asking for members from a deterministic backend is an
    error rather than a silent no-op: re-running a deterministic checkpoint
    would return identical copies, which would look like an ensemble while
    carrying no extra information.

    `variables` / `region` select which fields are kept and which box is
    cropped at write time. They are implemented for the anemoi (AIFS)
    backends only; asking for them on an Earth2Studio backend is an error
    rather than being silently ignored, because the caller would then get
    a global, all-variable zarr while believing it was regional."""
    name = name.lower()
    if name not in BACKEND_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BACKEND_FACTORIES)}")
    backend = BACKEND_FACTORIES[name]()

    if n_members is not None and n_members != 1:
        if not hasattr(backend, "n_members"):
            raise ValueError(
                f"Model '{name}' does not support ensemble members "
                "(only the anemoi ensemble checkpoints sample stochastically; "
                "re-running a deterministic model would give identical copies)."
            )
        backend = dataclasses.replace(backend, n_members=n_members)

    if variables or region:
        if not (hasattr(backend, "variables") and hasattr(backend, "region")):
            raise ValueError(
                f"Model '{name}' does not support --variables/--region yet "
                "(implemented for the anemoi AIFS backends only). Run it "
                "without them and crop afterwards."
            )
        from .regions import parse_region
        backend = dataclasses.replace(
            backend,
            variables=list(variables) if variables else None,
            # parsed here, not at run time: a bad region name should fail
            # before a 48 GB checkpoint is loaded
            region=parse_region(region) if region else None,
        )
    return backend