"""
Constant (time-invariant) surface fields AIFS requires as input: land-sea
mask (lsm), surface orography as geopotential (z), and the standard
deviation and slope of the sub-gridscale orography (sdor, slor).

anemoi's low-level SimpleRunner does NOT inject these from the checkpoint --
they must be present in the input state or prepare_input_tensor raises
"Missing variables in input fields: ['z', 'lsm', 'sdor', 'slor']".

These live in the SAME Arraylake ERA5 repo used for initial conditions, but
stored as 2-D COORDINATES on the single-level group rather than as data
variables -- which is why a data_vars-only lookup reports them missing. They
are present in both the free and the paid copy of the repo, so no extra
subscription, no CDS account and no second data source is needed.

Note the repo spells the orography `z_sfc`. Bare `z` in anemoi-land is this
surface field, while `z_500` etc are pressure-level geopotential -- two
different things that must not be conflated.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# anemoi field name -> candidate names in the ERA5 repo (coords or vars)
STATIC_ALIASES: dict[str, list[str]] = {
    "z": ["z_sfc", "z", "geopotential"],
    "lsm": ["lsm", "land_sea_mask"],
    "sdor": ["sdor", "standard_deviation_of_orography"],
    "slor": ["slor", "slope_of_sub_gridscale_orography"],
}
STATIC_FIELDS = STATIC_ALIASES  # anemoi_backend only needs the key set


def load_statics(source) -> dict[str, xr.DataArray]:
    """Pull the constant fields off an already-open ArraylakeERA5Source.

    Takes the source rather than opening anything itself: the repo is
    already open for initial conditions, the fields are on its single-level
    group, and re-opening would cost another set of round trips for data we
    are holding a handle to.

    Returns {anemoi_name: DataArray(lat, lon)}, grid-standardised the same
    way the source standardises everything else (descending lat, 0-360 lon)
    so it lines up with the time-varying fields."""
    ds = source.single
    available = list(ds.data_vars) + list(ds.coords)

    out: dict[str, xr.DataArray] = {}
    missing: dict[str, list[str]] = {}
    for name, candidates in STATIC_ALIASES.items():
        match = next((c for c in candidates if c in available), None)
        if match is None:
            missing[name] = candidates
            continue
        da = ds[match]
        # these are constant, but the stored array may still carry a
        # degenerate time-ish dim depending on how it was written
        for dim in ("time", "valid_time", "step", "number"):
            if dim in da.dims:
                da = da.isel({dim: 0}, drop=True)
        out[name] = source._fix_grid(da).load()
        logger.info("Static field %s <- %s %s", name, match, dict(out[name].sizes))

    if missing:
        raise KeyError(
            f"Constant field(s) not found in the ERA5 repo: {missing}. "
            "AIFS requires all of lsm/z/sdor/slor in its input state. They "
            "are normally 2-D coordinates on the single-level group -- "
            f"available names: {sorted(available)}"
        )
    return out