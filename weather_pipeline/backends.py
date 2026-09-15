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


def _atlas_backend() -> ModelBackend:
    from earth2studio.models.px import Atlas

    return ModelBackend("atlas", Atlas.load_default_package, Atlas.load_model)


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
    "atlas": _atlas_backend,
    "aifs1.0": lambda: _aifs_anemoi("1.0"),
    "aifs1.1": lambda: _aifs_anemoi("1.1"),
    # Ensemble checkpoint, but n_members=1: real ensemble members require
    # perturbed initial conditions (ECMWF ENS stream), which the ERA5
    # source cannot supply -- see the guard in AnemoiBackend.run.
    "aifs-ens1.0": lambda: _aifs_anemoi("ens1.0", n_members=1),
}


def get_backend(name: str) -> ModelBackend:
    name = name.lower()
    if name not in BACKEND_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BACKEND_FACTORIES)}")
    return BACKEND_FACTORIES[name]()