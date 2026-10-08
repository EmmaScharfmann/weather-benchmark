"""
Build the tutorial data pack from forecast zarrs + cached stations, and
optionally push it to a Hugging Face dataset repo.

    uv run python scripts/export_datapack.py \
        --zarr outputs/france_aug2023/aifs-ens1.0_*.zarr --region france \
        --cache outputs/cache/stations --out outputs/datapack_2023 --dev

    # then, after looking at the folder and its README.md:
    uv run python scripts/export_datapack.py ... --push you/aifs-france-downscaling-dev

Pushing creates the repo PRIVATE unless --public is given. --dev adds the
in-sample warning to the dataset card; use it for any period the model may
have been trained or fine-tuned on (summer 2023 for AIFS-ENS v1).

Needs `huggingface-cli login` (or HF_TOKEN) only for --push.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import xarray as xr

from weather_pipeline import station_interp as si
from weather_pipeline.datapack import export_pack
from weather_pipeline.regions import parse_region
from weather_pipeline.stations import MeteoFranceHourly, load_stations

DEV_CAVEAT = ("DEVELOPMENT DATA: these dates may fall inside the model's training or fine-tuning ")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", nargs="+", required=True)
    p.add_argument("--region", default="france")
    p.add_argument("--cache", default="outputs/cache/stations")
    p.add_argument("--out", required=True)
    p.add_argument("--variable", default="t2m")
    p.add_argument("--model", default="aifs-ens1.0")
    p.add_argument("--model-repo", default="ecmwf/aifs-ens-1.0")
    p.add_argument("--dev", action="store_true", help="add the development da")
    p.add_argument("--caveat", action="append", default=[], help="extra caveat line (repeatable)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--push", metavar="REPO_ID", default=None, help="push the folder to this HF dataset repo")
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("wetterdienst").setLevel(logging.WARNING)

    zarrs = sorted(a.zarr)
    first, last = xr.open_zarr(zarrs[0]), xr.open_zarr(zarrs[-1])
    start = pd.Timestamp(first.time.values[0]).floor("h")
    end = (pd.Timestamp(last.time.values[-1]) + pd.Timedelta(last.lead_time.values[-1])).floor("h")

    stations, _ = load_stations(MeteoFranceHourly(), parse_region(a.region), start, end,
                                variables=[a.variable], cache_dir=a.cache)

    from weather_pipeline.datasource import build_data_source
    from weather_pipeline.statics import load_statics
    era5 = build_data_source("era5")
    orography = si.height_from_geopotential(load_statics(era5)["z"])

    caveats = ([DEV_CAVEAT] if a.dev else []) + a.caveat
    out = export_pack(zarrs, stations, orography, era5, a.out, a.variable, a.model, a.model_repo,
                      caveats=caveats, overwrite=a.overwrite)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"pack written: {out} ({size:.0f} MB). Read {out}/README.md before pushing.")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="dataset", private=not a.public, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push, repo_type="dataset",
                          commit_message=f"data pack {a.model} {a.variable} {start:%Y-%m-%d}..{end:%Y-%m-%d}")
        print(f"pushed to https://huggingface.co/datasets/{a.push} "
              f"({'public' if a.public else 'private'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())