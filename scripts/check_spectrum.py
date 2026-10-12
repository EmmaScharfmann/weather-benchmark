"""Compare the spatial power spectra of the coarse field, the model and the target.

Usage::

    python scripts/check_spectrum.py --config config.yaml --model runs/crps/model.pt

Answers whether the model adds detail at the scales the target actually has.
Power well below the target's at short wavelengths means the field is too
smooth; power well above means the model is inventing texture the target does
not contain. The ensemble mean is expected to sit below the target, since
averaging removes unpredictable detail; an individual member is the one that
should match.

Outputs in the model's directory: ``spectrum.csv``, ``spectrum.png`` and a
printed summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from downscale.config import load_config
from downscale.diagnostics import (
    effective_resolution_km,
    grid_spacing_km,
    spectrum_comparison,
    spectrum_figure,
)
from downscale.downscaler import Downscaler

logger = logging.getLogger("spectrum")
LABELS = ["target", "coarse", "model member", "model mean"]


def validation_slice(pairs_dir: Path, val_years: Sequence[int], n_times: int, hour: int | None, seed: int):
    """Select a sample of validation times from the training pairs.

    :param pairs_dir: Directory written by the pairing stage.
    :type pairs_dir: pathlib.Path
    :param val_years: Years held out of training.
    :type val_years: collections.abc.Sequence[int]
    :param n_times: Number of times to draw.
    :type n_times: int
    :param hour: Restrict to this UTC hour; ``None`` uses every hour.
    :type hour: int | None
    :param seed: Random seed for the draw.
    :type seed: int
    :returns: Coarse fields, target fields and their times.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If no time matches even after dropping the year filter.
    """
    store = pairs_dir / "pairs.zarr"
    ds = xr.open_zarr(store) if store.exists() else xr.open_mfdataset(
        [str(p) for p in sorted((pairs_dir / "months").glob("pairs_*.zarr"))],
        engine="zarr", combine="nested", concat_dim="time",
    )
    times = pd.DatetimeIndex(ds.time.values)
    chosen = times.year.isin(list(val_years)) if val_years else np.ones(len(times), bool)
    if not chosen.any():
        logger.warning(
            "no time falls in the validation years %s: sampling every year instead, so the spectrum "
            "includes data the model was trained on",
            list(val_years),
        )
        chosen = np.ones(len(times), bool)
    if hour is not None:
        chosen = chosen & (times.hour == hour)
    index = np.flatnonzero(chosen)
    if index.size == 0:
        raise ValueError(f"no time matches hour {hour}")
    picked = np.sort(np.random.default_rng(seed).choice(index, min(n_times, index.size), replace=False))
    subset = ds.isel(time=picked)
    return (
        subset.coarse_t2m.values.astype("f4"),
        subset.fine_t2m.values.astype("f4"),
        times[picked].to_numpy(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    :param argv: Argument list; ``None`` reads ``sys.argv``.
    :type argv: collections.abc.Sequence[str] | None
    :returns: Exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", default=None, help="overrides paths.pairs")
    parser.add_argument("--out", default=None, help="defaults to the model's directory")
    parser.add_argument("--n-times", type=int, default=24)
    parser.add_argument("--hour", type=int, default=12, help="UTC hour to sample; -1 uses every hour")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    pairs_dir = Path(args.pairs or cfg.paths.pairs)
    out = Path(args.out) if args.out else Path(args.model).parent
    out.mkdir(parents=True, exist_ok=True)

    coarse, target, times = validation_slice(
        pairs_dir, cfg.val_years, args.n_times, None if args.hour < 0 else args.hour, args.seed
    )
    model = Downscaler.load(args.model, args.device)
    predicted = model.predict(coarse, times, members=args.members)
    spacing = grid_spacing_km(model.lat, model.lon)
    logger.info(
        "%d time(s), grid spacing %.2f km, Nyquist wavelength %.1f km", len(times), spacing, 2 * spacing
    )

    comparison = spectrum_comparison(
        {
            "target": target,
            "coarse": coarse,
            "model member": predicted[:, 0],
            "model mean": predicted.mean(1),
        },
        spacing_km=spacing,
    )
    comparison.to_csv(out / "spectrum.csv", index=False)
    spectrum_figure(comparison, LABELS, out / "spectrum.png")

    summary = {
        "n_times": len(times),
        "grid_spacing_km": round(spacing, 3),
        "effective_resolution_km": {
            name: round(effective_resolution_km(comparison, name, "target"), 1)
            for name in ("coarse", "model member", "model mean")
        },
    }
    short = comparison[comparison.wavelength_km < 10 * spacing]
    summary["power_ratio_below_%dkm" % int(10 * spacing)] = {
        name: round(float((short[name] / short["target"]).mean()), 3)
        for name in ("coarse", "model member", "model mean")
    }
    (out / "spectrum.json").write_text(json.dumps(summary, indent=2))

    bands = [(400, np.inf), (200, 400), (100, 200), (50, 100), (25, 50), (2 * spacing, 25)]
    rows = []
    for low, high in bands:
        chosen = comparison[(comparison.wavelength_km >= low) & (comparison.wavelength_km < high)]
        if not len(chosen):
            continue
        label = f"{low:.0f}-{high:.0f} km" if np.isfinite(high) else f"above {low:.0f} km"
        rows.append({
            "wavelengths": label,
            **{name: round(float((chosen[name] / chosen["target"]).mean()), 3)
               for name in ("coarse", "model member", "model mean")},
        })
    print("\npower relative to the target (one means a perfect match)")
    print(pd.DataFrame(rows).set_index("wavelengths").to_string())
    print("\nsmallest wavelength still resolved, at half the target's power:")
    for name, value in summary["effective_resolution_km"].items():
        print(f"  {name:14s} {value} km")
    member_ratio = summary[f"power_ratio_below_{int(10 * spacing)}km"]["model member"]
    if member_ratio > 1.5:
        print(
            f"\nthe member carries {member_ratio:.1f} times the target's power below "
            f"{int(10 * spacing)} km: the model is adding detail the target does not have"
        )
    elif member_ratio < 0.5:
        print(f"\nthe member is too smooth below {int(10 * spacing)} km ({member_ratio:.2f} of the target)")
    else:
        print(f"\nthe member matches the target below {int(10 * spacing)} km ({member_ratio:.2f})")
    print(f"wrote {out}/spectrum.csv, spectrum.png, spectrum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())