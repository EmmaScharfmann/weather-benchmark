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
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run_deterministic

from .datasource import ArraylakeERA5Source

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


def _aifs_v1_0_package():
    """AIFS.load_default_package() in earth2studio 0.18.0 is hardcoded to
    v1.1 -- there's no version kwarg in this release. To get v1.0
    specifically, build the Package manually the same way earth2studio's
    own load_default_package() does, just pointed at the older HF repo.
    Unverified beyond "it should work" -- AIFS derives its variable
    ordering from each checkpoint's own ai-models.json metadata, so this
    *should* generalize, but test it before trusting the output."""
    from earth2studio.models.auto.package import Package

    return Package(
        "hf://ecmwf/aifs-single-1.0",
        cache_options={
            "cache_storage": Package.default_cache("aifs-single-1.0"),
            "same_names": True,
        },
    )


def _aifs_backend(version: str | None = None) -> ModelBackend:
    from earth2studio.models.px import AIFS
    if version  == "1.1":
        name = "aifs1.1"
        return ModelBackend(name, AIFS.load_default_package, AIFS.load_model)
    elif version == "1.0":
        return ModelBackend("aifs1.0", _aifs_v1_0_package, AIFS.load_model)
    else:
        raise ValueError(f"Unsupported AIFS version: {version!r}")


def _aifs2_backend() -> ModelBackend:
    from earth2studio.models.px import AIFS2

    return ModelBackend("aifs2", AIFS2.load_default_package, AIFS2.load_model)


def _atlas_backend() -> ModelBackend:
    from earth2studio.models.px import Atlas

    return ModelBackend("atlas", Atlas.load_default_package, Atlas.load_model)


BACKEND_FACTORIES: dict[str, Callable[[], ModelBackend]] = {
    "aurora": _aurora_backend,
    "aifs1.1": lambda: _aifs_backend(version="1.1"),      # explicit alias for "aifs" -- same checkpoint
    "aifs1.0": lambda: _aifs_backend(version="1.0"),      # manually points at the older v1.0 HF repo -- verify before trusting
    "aifs2": _aifs2_backend,                              # AIFS v2.0 (waves, 10 hPa level, snow cover)
    "atlas": _atlas_backend,
}


def get_backend(name: str) -> ModelBackend:
    name = name.lower()
    if name not in BACKEND_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BACKEND_FACTORIES)}")
    return BACKEND_FACTORIES[name]()