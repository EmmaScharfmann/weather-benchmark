# Do we really need bigger AI weather forecasting models to improve resolution?

## TL;DR

In this blog, I want to share something NLP has taught me: **you don’t always need a bigger model; often you need to adapt the model to your task. The same idea applies to AI weather forecasting.** If you need forecasts at higher resolution, you don’t necessarily need a bigger model at that resolution.. You can take an existing global model and train a downscaler on top of it to learn the local details, like the effects of orography or coastlines. For many applications, this gives you a more locally accurate forecast that is much cheaper to train and run than a high-resolution global model

[ Image with coarse model VS higher resolution model ]

## Table of Contents

 - [Intro](#Intro)
- [What is a downscaler & demo](#what-is-a-downscaler)
- [Downscaler's components](#downscalers-components)
- [Example: training a downscaler to predict the temperature over France at 0.1°](#example-training-a-downscaler-to-predict-the-temperature-over-france-at-01°)
- [Train your downscaler on Hugging Face]
- [Further reading]


## Intro

AI weather models are very good at forecasting large-scale weather patterns. But they see the world as a grid, often at 0.25° resolution (roughly equivalent to a grid of 25 × 25 km), where each cell holds a single value for temperature, wind, and every other variable. Anything smaller than a cell gets averaged out, so local effects from valleys, mountains, islands, or coastlines are lost.

Big labs are tackling this by building models that work at finer resolution. WeatherNext 3, for example, forecasts surface temperature and humidity at around 5 km. But building these models requires the data, compute, and engineering that only a few organizations have.

If NLP has taught me anything, it’s that you don’t always need a bigger model; often you need to adapt an existing model to your task. The same idea applies to AI weather forecasting: instead of retraining global models at higher resolution, we can adapt existing ones to the region, variables and resolution we care about.

One way to do this is to train a downscaler on top of an existing model to learn the local details. It’s cheap: I trained one in a few hours on a single GPU to predict temperature over France, and it improved AIFS Ensemble 1.0 forecasts by X% against observations (temperature) from weather stations.

## What is a downscaler? 

A weather forecasting model works at a specific resolution: it only knows the weather on the cells of a grid and is blind inside each cell. A downscaler is an ML model that takes the low resolution forecast and fills in the local details. In NLP terms, it’s like adding a task-specific head on top of a frozen embedding model: the big model provides a good general representation of the atmosphere, and the small head adapts it to what you need.

To train a downscaler, you need pairs of the same weather at two resolutions: a coarse field (low resolution), like the output of a global model, and a fine field, like a high-resolution regional reanalysis. The downscaler learns the difference between the two, using static information such as orography and land–sea boundaries to learn how the weather varies within a coarse cell: a mountaintop is colder than the valley below, and the coast is milder than inland. Once trained, you can plug it on top of a coarse forecast model and get a high-resolution forecast at low cost.

![Screenshot 2026-10-12 at 12.05.44](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/Q8zMy0TE5pZ6CIYwc6aaJ.png)

However, like all ML models, downscalers are unfortunately not magical. They perform well on some tasks and poorly on others. So before training one, we should make sure that what we are trying to improve can be improved by a downscaler. Below is my summary what a downscaler can and cannot improve.
| what a downscaler can imporove | what a downscaler can't impove |
|---|---|
| Add the patterns due to the terrain (valleys, mountains, coastlines) | Fix global errors in the coarse model: if the coarse forecast puts the heat in the wrong region, the downscaler will not fix it |
| Produce a single or an ensemble of fine fields from one coarse field | Add details that the inputs do not contain (it cannot add the effect of mountains if you do not input the elevation) |
| Work behind any coarse model (as it never sees a forecast during training) | Reduce the uncertainty of the large-scale forecast |
| Run in seconds, once trained | Exceed the quality of the fine field used during training |



[ Demo ]

## Downscaler's components

### Training 

Training a downscaler involves 4 elements.

1. **Two datasets**: a coarse dataset and a fine dataset, covering the same place and the same times.
  - In our example, for the coarst dataset, we used ERA5 at 0.25°.
  - In our example, for the fine dataset, CERRA at 5.5 km was used.
Note that we chose those two datasets because CERRA takes its boundary conditions from ERA5, so they describe the same weather at two resolutions. For every time step, we build a pair of the coarse field interpolated onto the fine grid and the fine field itself. The interpolated field is used as the model’s main input and as the baseline it has to beat. The model is trained to learn to predict the difference between the coarse field and the fine field.

2. **Static data**: In addition to the field to predict, the downscaler needs additional information to infer what is happening within the cell. These fields never change, which is exactly why the model learns easily from them: the correction they imply is the same on every day. We add the season and the time of the day, so the model can also learn how that correction varies through the day and through the year. In our example, we’re using the elevation of the fine-scale field, the height difference between the real surface and the coarse model’s surface, the land fraction of the cells, the longitude/latitude and the season and time of the day as static data.

4. **Loss function**: We made the choice in our example of building a downscaler for an ensemble model. So, we use fair CRPS, which scores an ensemble rather than a single field. It rewards being close to the truth and having realistic spread. Mean squared error could have been used as a loss function if building a downscaler for a deterministic model. Note that the loss in our example is computed over land only, since the target reanalysis produces sea surface temperature by a different route.

5. **Downscaler**: it is a deep learning model which learns to predict the difference between the coarse field and the fine field from the coarse field and the static data. In our example, we trained a U-Net with about 2M weights, which took around an hour to train on a GPU.

![downscaler_training_diagram](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/ThaGcN9zWdbnWuSLAs4Is.png)

### Inference

**Input data**: the downscaler takes the coarse ensemble forecast as input, as well as the same static data used during training, which is stored inside the model. Each coarse member is interpolated onto the fine grid before the forward pass. In our example, ten members from AIFS-ENS at 0.25° are used as input.

**Downscaler**: it takes the interpolated coarse field, the static data and a draw of noise, then predicts the correction to add to the coarse field. Since the noise is drawn  on each pass, running the same member several times gives multiple fine fields. In our example we draw four samples per member.

**Output**: the model outputs multiple members at fine resolution. In our example, each of the ten coarse members yields four samples, giving forty members at 5.5 km. Note that in our example, running the downscaler takes only a few seconds per forecast.

![downscaler_inference](https://cdn-uploads.huggingface.co/production/uploads/6a2f0f8a8aec62a63aae7447/nrYkMy6CfIv91mUhlJx5W.png)

## Example: training a downscaler to predict the temperature over France at 0.1°

## How to use the downscaler

_Before you start_: You will also need a free Earthmover account ([sign up here](https://app.earthmover.io/auth/login)). 

0. Set up
Clone the repo and install dependencies. 
  ```bash
git clone https://github.com/EmmaScharfmann/weather-forecast-model-downscaler
cd weather-forecast-model-downscaler
uv sync
uv pip install torch 
```

The coarse fields are fetched from Earthmover. The code uses the Arraylake client which requires a free Earthmover account. You can identify by running: 
```
uv run arraylake auth login
# or
export ARRAYLAKE_TOKEN=ema_xxxxxxxxxxxx
```

1. Fetch the coarse field

```python
import pandas as pd
from downscale.datasource import build_data_source

era5 = build_data_source("era5")
times = pd.date_range("2020-08-01", "2020-08-31T18:00", freq="6h")
coarse = era5(list(times), ["t2m"]).sel(variable="t2m", drop=True)
```

2. Run the downscaler

The model needs a fine grid as input, so the coarse field has to be interpolated first. The static data are part of the model's checkpoint on the Hub. Here, we choose to predict 4 fine members from the coarse field. 

```python
from huggingface_hub import hf_hub_download
from downscale.downscaler import Downscaler
from downscale.station_interp import GridInterpolator

model = Downscaler.load(hf_hub_download("EmmaScharfmann/france-temperature-downscaler", "model.pt"))
interp = GridInterpolator.build(coarse.lat, coarse.lon, model.lat.ravel(), model.lon.ravel())
on_fine = interp.apply(coarse).values.reshape(len(times), *model.lat.shape)
fine = model.predict(on_fine, times.to_numpy(), members=4)
```
`fine` is the output of the downscaler: its share is `(time, 4, y, x)` and corresponds to the temperature in Kelvin of France at a resolution of 0.1°. On a GPU, the downscaler takes seconds to run. 

3. Compare forecasts against CERRA

CERRA is what the model was learned to reproduce during the training. Comparing against it shows how much the model actually learned.

```python
import xarray as xr

USE_CACHE = True  # False to download the data from CERRA and read it locally

if USE_CACHE:
    cerra = xr.open_zarr("hf://datasets/EmmaScharfmann/cerra-france-t2m-2016-2020/target.zarr")
else:
    from huggingface_hub import snapshot_download
    folder = snapshot_download("EmmaScharfmann/cerra-france-t2m", repo_type="dataset")
    cerra = xr.open_zarr(f"{folder}/target.zarr")

truth = cerra.t2m.sel(time=times).values
for name, field in [("coarse", on_fine), ("downscaled", fine.mean(1))]:
    print(name, np.sqrt(((field - truth) ** 2).mean()).round(3), "K")
```

4. Compare forecasts against observations

Oberservations from Meteo France weather stations is used to evaluation the model against an independant ground truth. Downloading the station data takes around 1 hour for 1 month, so we cached the stations' data of August 2020 on the Hub to save some time. You can download weather stations from Meteo France by setting `USE_CACHE=False` below.

```python
import numpy as np, xarray as xr
from downscale.stations import MeteoFranceHourly, load_stations
from downscale.station_interp import curvilinear_for

USE_CACHE = True # False to download weather stations directly from the Meteo France API. 

# Downloading the observations
if USE_CACHE:
    stations = 
else:
    stations, _ = load_stations(MeteoFranceHourly(), "france", times[0], times[-1])
read = curvilinear_for(model.lat, model.lon, stations).apply
obs = stations.t2m.sel(time=times).values
for name, field in [("coarse", on_fine), ("downscaled", fine.mean(1))]:
    at_stations = read(xr.DataArray(field, dims=("time", "y", "x"))).values
    print(name, np.sqrt(np.nanmean((at_stations - obs) ** 2)).round(3), "K")
```





You can train your own downscaler, either the same way as we did, or by adapting some of the components. 
The code can be found here: [Downscaler code]()
The data can be found here: [Cerra data from 2016 to 2020](), [Weather stations data]() 
The downscaler can be found here: [Downscaler]() 




## Further Reading 

## Technical details 

Here are a few technical details that we think are worth sharing, from the experiments we ran:
- the downscaler predicts the correction between the coarse and the fine field instead of the temperature. Therefore, the training begins from the baseline rather than from noise.
- the downscaler takes noise channels to produce several plausible fine fields from one coarse field. Each forward pass draws noise. The noise is drawn at several spatial scales to avoid the ensemble’s variability piling up at a given wavelength.
- the downscaler upsamples by interpolating then convolving, rather than with transposed convolutions, to avoid producing a checkerboard pattern at the two-pixel scale.
