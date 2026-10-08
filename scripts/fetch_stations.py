import argparse
import pickle
from pathlib import Path
from weather_pipeline.stations import MeteoFranceHourly, load_stations

parser = argparse.ArgumentParser(description="Fetch Météo-France hourly station data")
parser.add_argument("start", help='Start date, e.g. "2025-08-01"')
parser.add_argument("end", help='End date, e.g. "2025-08-30 23:00"')
parser.add_argument("--variables", nargs="+", default=["t2m", "tx1h"])
parser.add_argument("--outdir", default="outputs")
args = parser.parse_args()

tag = f"{args.start[:10]}_{args.end[:10]}".replace("-", "")
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

ds, report = load_stations(
    MeteoFranceHourly(), "france", args.start, args.end,
    variables=args.variables, cache_dir=str(outdir / "cache" / f"stations_{tag}"),
)

ds.to_netcdf(outdir / f"stations_{tag}.nc")
with open(outdir / f"stations_{tag}_report.pkl", "wb") as f:
    pickle.dump(report, f)
print(f"Done: {outdir / f'stations_{tag}.nc'}")