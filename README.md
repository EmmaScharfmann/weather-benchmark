# weather-forecast-model-downscaler

Train a downscaler that turns a 0.25° ensemble temperature forecast into 5.5 km maps over France, and
apply it to AIFS Ensemble 1.0 forecasts read from Earthmover.

The downscaler is a small U-Net trained on pairs of ERA5 (coarse) and CERRA (fine). It learns the
correction from the interpolated coarse field to the fine one, from the terrain, the coast, the time of
day and of year, and a draw of noise, so every coarse member yields several plausible fine maps.
Running the forecast models themselves lives in a separate project,
[earth2studio-pipeline](https://github.com/EmmaScharfmann/earth2studio-pipeline).

## Setup

```bash
uv sync
uv run arraylake auth login      # Earthmover account: ERA5 and AIFS-ENS
uv run hf auth login             # model and datasets on the Hugging Face Hub
```

AIFS-ENS comes from the [Earthmover marketplace](https://app.earthmover.io/marketplace/6a10729e8a69e6bcaeae0f90).
Subscribing creates a repository under your organisation; set `forecasts.source` in `config.yaml` to it.

## Tutorial

`tutorial.ipynb` fetches an AIFS-ENS run, applies the published downscaler
([EmmaScharfmann/france-temperature-downscaler](https://huggingface.co/EmmaScharfmann/france-temperature-downscaler)),
maps both forecasts, and evaluates them against CERRA, CERRA's own floor and Météo-France stations.

## Pipeline

Every script takes `--config config.yaml`; command-line flags override it.

| Step | Script |
|---|---|
| Download the CERRA target (Copernicus account) | `scripts/download_cerra.py` |
| Pair it with ERA5 from Earthmover | `scripts/build_pairs.py`, then `--consolidate` |
| Train | `scripts/train_downscaler.py --out runs/crps` |
| Check the model against its target's floor | `scripts/check_target_floor.py --model runs/crps/model.pt --start ... --end ...` |
| Check the power spectrum | `scripts/check_spectrum.py --model runs/crps/model.pt` |
| Fetch AIFS-ENS forecasts | `scripts/fetch_forecasts.py --start 2025-08-01 --end 2025-08-31` |
| Downscale them | `scripts/downscale_forecasts.py --model runs/crps/model.pt` |
| Score both against stations | `scripts/eval_downscaled.py` |
| Publish the model or datasets to the Hub | `scripts/publish.py` |

## Layout

- `downscale/`: the library. `sources.py` (ERA5 and AIFS-ENS access), `downscaler.py` (network and
  checkpoint), `inference.py` (apply to a forecast), `evaluation.py` and `plots.py` (scores and figures
  for the tutorial), `stations.py` and `station_scoring.py` (observations and scoring), `diagnostics.py`
  (report figures), `fetch.py` and `publish.py` (Hub datasets).
- `scripts/`: the pipeline steps above.
- `tests/`: `uv run pytest`, on synthetic data, no network needed.
- `findings.md`: behaviour of the data and libraries worth knowing.
