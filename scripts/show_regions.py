"""
Show regionally-binned scores: models x regions x lead time.

Reads the '#binned' states written by `run_benchmark.py --bin-regions`.
Those carry an extra `region` dimension that the HTML report does not
handle, so they get their own (much simpler) viewer here.

The question this answers directly: is a model's error concentrated
somewhere, or spread evenly? A global RMSE cannot tell you whether a model
is uniformly mediocre or excellent everywhere except the Arctic.

Usage:
    uv run scripts/show_regions.py --variable t2m --metric rmse
    uv run scripts/show_regions.py --variable z500 --metric rmse --lead 48
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from weather_pipeline import wbx_scoring as wbx

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
# Binned states live in their own directory: a filename marker is not an
# option because zarr/fsspec truncate paths at "#".
SCORES_DIR = REPO_ROOT / "outputs/scores/binned"

_RE = re.compile(
    r"^(?P<model>.+)_(?P<tag>[^_]+)_(?P<init>\d{8}T\d{4})_state\.zarr$"
)


def load(variable: str) -> dict[str, tuple[list, bool]]:
    out: dict[str, tuple[list, bool]] = {}
    if not SCORES_DIR.exists():
        return out
    for path in sorted(SCORES_DIR.glob("*_state.zarr")):
        m = _RE.match(path.name)
        if not m or m.group("tag") != variable:
            continue
        state, ens = wbx.load_state(path)
        states, was_ens = out.get(m.group("model"), ([], False))
        states.append(state)
        out[m.group("model")] = (states, was_ens or ens)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable", default="t2m")
    p.add_argument("--metric", default="rmse")
    p.add_argument("--lead", type=int, default=None,
                   help="Lead time in hours. Default: the longest available.")
    args = p.parse_args()

    states = load(args.variable)
    if not states:
        raise SystemExit(
            f"No binned states for '{args.variable}' in {SCORES_DIR}. "
            "Run the benchmark with --bin-regions first."
        )

    rows = {}
    for model, (model_states, ensemble) in sorted(states.items()):
        metrics = wbx.build_metrics(ensemble=ensemble)
        # States are SUMMED before metric values are computed -- the same
        # correct aggregation the global report uses.
        results = wbx.combine_states(model_states, metrics)
        col = f"{args.metric}.{args.variable}"
        if col not in results:
            logger.warning("%s has no %s; skipping", model, col)
            continue

        df = results[col].to_dataframe().reset_index()
        df["lead_h"] = (
            pd.to_timedelta(df["lead_time"]).dt.total_seconds() / 3600
        ).astype(int)
        lead = args.lead if args.lead is not None else df["lead_h"].max()
        sel = df[df["lead_h"] == lead]
        if sel.empty:
            logger.warning("%s has no lead time +%d h", model, lead)
            continue
        rows[model] = sel.set_index("region")[col]
        chosen = lead

    if not rows:
        raise SystemExit("Nothing to show.")

    table = pd.DataFrame(rows)
    print(f"\n{args.metric} for {args.variable} at +{chosen} h "
          f"({len(next(iter(states.values()))[0])} init time(s))\n")
    print(table.round(4).to_string())

    # Ratio to the global-ish reference region makes the pattern legible:
    # a model twice as bad in the Arctic as the tropics shows up as 2.0
    # regardless of the variable's units.
    if "tropics" in table.index:
        print("\nrelative to tropics:")
        print((table / table.loc["tropics"]).round(2).to_string())


if __name__ == "__main__":
    main()