# earth2studio-pipeline

Run Aurora / AIFS / AIFS2 / Atlas on Arraylake ERA5 initial conditions via Earth2Studio,
with shared postprocessing, RMSE scoring, and rotating-globe visualization.

## Setup

```bash
uv venv
source .venv/bin/activate

# install torch first (match your CUDA version), then the model(s) you need
uv pip install torch
uv sync --extra aifs           # or --extra aurora / --extra aifs2 / --extra atlas / --extra all-models
uv sync --extra notebook        # optional, for Jupyter
```

## Run

```bash
uv run run-forecast --model aifs --init-time 2026-03-13T00:00:00 --n-steps 8
# equivalently: uv run python -m earth2studio_pipeline.pipeline --model aifs ...
```

Or from Python / a notebook, see `example_usage.py`.