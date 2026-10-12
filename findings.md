# Findings

Behaviour of the data, the libraries and the models that is not visible from
the code, recorded while building this pipeline. Each entry names the file it
affects. Dates refer to when the behaviour was observed.

## Grids and coordinates

**Cropped regions are not contiguous in storage order.**
`weather_pipeline/regions.py`, `weather_pipeline/station_interp.py`

Gridded archives store longitudes 0..360 with descending latitudes. A region
spanning the prime meridian, such as France at 7°W to 11°E, becomes two blocks:
353..359.75 and 0..11. `crop_indices` returns longitude indices ordered west to
east, so the resulting coordinate reads 353, …, 359.75, 0, …, 11 and is **not
monotonic**. This keeps the array compatible with the archives' own convention,
but anything that interpolates in longitude must convert with `to_signed_lon`
first.

**The regional reanalysis uses a Lambert grid.**
`weather_pipeline/station_interp.py`, `scripts/download_cerra.py`

CERRA's latitude and longitude are two-dimensional fields over `(y, x)`, so the
regular-grid interpolator does not apply. `CurvilinearInterpolator` uses a
k-d tree on three-dimensional unit vectors, which gives true great-circle
distances and needs no special handling at the meridian.

**A rotated fine grid pokes outside a rectangular coarse domain.**
`scripts/downscale_forecasts.py`

The corners of the cropped CERRA grid fall just outside the France box the
forecasts were cut to, about 0.2% of points. These take edge values with a
warning; more than `MAX_FRACTION_OUTSIDE` is treated as a misconfigured region.

**The Climate Data Store cannot crop a Lambert grid.**
`scripts/download_cerra.py`

Each request downloads the full European domain and the crop happens locally.
The index rectangle is computed once and saved to `crop.json`, so every month
shares exactly the same grid; a month whose source grid differs is rejected
rather than silently misaligned.

## Model output

**The forecast store starts at the first lead step, not at zero.**

A ten-day forecast at six-hourly steps holds 40 entries beginning at +6 h.
Label results using `lead_time` values rather than integer positions.

**Six-hourly output cannot resolve daily extremes.**

The warmest hour of the day usually falls between two steps, so daily maxima
computed from four samples are systematically too low. Score the six-hourly
values, or define events at the steps nearest the afternoon peak.

**Ensemble members from one initial state understate uncertainty.**

When every member starts from the same analysis, spread comes only from the
model's own sampling. Expect a spread-skill ratio well below one at short lead
times. Station verification lowers it further, because the roughly 2 K mismatch
between a grid cell and a point appears in the error but in no member.

## Station observations

**Station metadata is unreliable about history.**
`weather_pipeline/stations.py`

Of about 13,800 Météo-France stations listed inside the France box, roughly 680
have no start date. A sample of twelve returned no data at all for August 2023;
their names end in `-SPC`, suggesting a separate flood-forecasting network. They
are excluded by default, and `include_unknown_start` costs only download time
because the completeness check removes them anyway.

**Many listed stations do not measure temperature.**

For 5–30 August 2023, 2,229 stations were active over the whole window but 429
of them returned no temperature at all. They appear in the quality-control
report with zero completeness. The network of about 1,800 temperature stations
is what remains.

**Requesting a closed station fails the whole chunk.**
`weather_pipeline/stations.py`

The underlying service raises a missing-file error, which aborts every station
requested alongside it. The provider retries the chunk one station at a time.

**A corrupted download cache surfaces as an unrelated type error.**
`weather_pipeline/stations.py`

An interrupted `wetterdienst` download leaves a half-written entry in
`~/.cache/wetterdienst`, after which reads fail with
`TypeError: cannot unpack non-iterable bool object` inside `fsspec`. The
provider retries that chunk with the cache disabled; clearing the directory
fixes it permanently.

**The cache key is the window, not the period covered.**
`weather_pipeline/stations.py`

Caching per station and window means a request for a different window refetches
everything. `_covering_dirs` therefore reuses any cached window that fully
contains the requested one. A narrower request is free; a wider one must fetch.

## Verification

**A gridded reanalysis is a circular reference for temperature.**

ERA5-Land applies its own lapse-rate correction to ERA5, which also provides
the forecasts' initial conditions, so a lapse-rate baseline would nearly
reproduce it by construction. Station observations are the only independent
truth here, and the only one available after CERRA ends in mid-2021.

**Scores must be summed, not averaged, across initialisations.**
`weather_pipeline/station_scoring.py`

Combining must sum the per-case statistics and take the square root afterwards.
Averaging per-case root mean squared errors differed by 0.56 K in one test.

**Ensembles of different sizes cannot share a dimension name.**
`weather_pipeline/station_scoring.py`, `scripts/eval_downscaled.py`

Putting a 10-member and a 40-member forecast into one dataset under the name
`ensemble` makes xarray align them, padding the smaller with missing values and
silently producing missing ensemble scores. The downscaled ensemble therefore
uses `ensemble_ds`, and the scorer accepts any dimension whose name begins with
`ensemble`.

**A dense cluster of stations distorts the mean.**
`weather_pipeline/station_scoring.py`

With 40 of 50 synthetic stations crammed into one spot and biased by 2 K, the
unweighted mean bias was 1.61 K and the density-weighted one 0.24 K.

**Bootstrap intervals over a dozen initialisations are illustrative.**
`weather_pipeline/station_scoring.py`

Neighbouring initialisations share weather, which is why blocks are resampled
together. With few cases the intervals describe sampling noise within the
period studied, not uncertainty about the method in general.

## Training

**A constant lapse rate helps by day and harms at night.**

Measured against about 1,800 stations over an August heatwave, the correction
reduced the error by 0.4–0.55 K at 12 and 18 UTC and slightly increased it at
00 and 06 UTC, when cold air pools in valleys. By elevation, it moved the bias
above 2,500 m from +4.91 K to −0.47 K while worsening the lowlands. The overall
average looked unfavourable only because most stations are low.

**The coarse grid damps the daily cycle.**

Interpolated fields are warm at night and cold at midday relative to stations,
for the analysis as much as the forecast. No height correction can fix this,
since it reflects the difference between a cell average and a point.

**Mean squared error and fair CRPS fail differently.**
`weather_pipeline/downscaler.py`

On synthetic data with a known noise floor, the mean squared error model
reached the best possible root mean squared error with zero spread, while the
CRPS model reached the theoretical best score of 0.564 K with a spread-skill
ratio approaching one. The first gives one smooth map; the second keeps
realistic variability.

**Masking the sea in the loss changes the gradient but not always the result.**
`weather_pipeline/downscaler.py`, `scripts/train_downscaler.py`

With `training.land_only`, the sea is still given to the network as input, since
coastal gradients carry information, but it is excluded from the loss and from
the normalisation statistics. The unit tests confirm the excluded region
receives no gradient. On synthetic data whose sea target was pure noise, the
land scores were identical either way (0.301 K both), because zero-mean noise
does not pull the shared weights in any particular direction. The setting
should matter when the real sea target carries learnable structure that
competes with the land signal, which is worth testing by training both ways and
comparing the land scores.

**The output layer starts at zero.**
`weather_pipeline/downscaler.py`

Zero-initialising the final convolution makes the untrained model reproduce the
interpolated field exactly, so training begins from the baseline rather than
from noise.

**Noise is drawn at two scales.**
`weather_pipeline/downscaler.py`

Half the noise channels are full resolution and half are drawn at one eighth
and upsampled, so members differ coherently over tens of kilometres instead of
only pixel by pixel.

**Domain edges carry more spread.**

The network has less context at the boundary, which inflates the spread there.
It fades with training, and the stations sit well inside the domain.

## Dependencies

**Reading GRIB needs the ecCodes binary library.**

`pip install cfgrib` alone is not enough; the Python `eccodes` package must be
recent enough to pull in `eccodeslib`, which ships the library as a wheel. Pin
`eccodes>=2.38` and `eccodeslib` explicitly, and verify with
`python -m eccodes selfcheck`. The wheel needs glibc 2.28 or newer; on older
systems install ecCodes from conda-forge and set `ECCODES_DIR`.

**Pin the scoring library.**

WeatherBench-X is installed from its git repository, so pin a commit for
reproducibility.