# Future Space

Future Space collects public space-weather data from multiple providers and harmonizes it
into a single time-indexed dataset. It uses that dataset to forecast geomagnetic storms with
machine learning. Data, models, forecasts and their verification are published openly, with
no infrastructure cost.

> Main (Portuguese) version, with authorship, evaluation and citation: [README.md](README.md)

| Resource | Location |
|---|---|
| Website (PT, EN, ES, FR) | https://future-space-six.vercel.app |
| API documentation (OpenAPI) | https://future-space-six.vercel.app/docs |
| Dataset and models | https://huggingface.co/datasets/Strutf/future-space |
| Model history | https://future-space-six.vercel.app/v1/models |

The forecasts are experimental and must not be used for operational decisions. Official
forecasts are issued by [NOAA SWPC](https://www.swpc.noaa.gov/).

## Overview

A solar storm develops in stages, and each stage is observed by different instruments:

```
   SUN                  SPACE (L1)                EARTH ORBIT              GROUND
 flares, active   ──▶  solar wind and        ──▶  X-rays, energetic   ──▶  Kp and Dst indices,
 regions, coronal      magnetic field             protons                  magnetometers,
 mass ejections        (~1 h before Earth)                                 ionosphere / GPS
```

The project has three components:

1. **Data collection.** Fifteen datasets are collected every hour, standardized (UTC `time`
   column, consistent units and missing values) and archived. Several sources keep only 1–7
   days of real-time data, so continuous collection is what makes the archive possible.
2. **Forecasting.** Gradient-boosted tree models forecast the Kp index (3–24 h ahead) and the
   Dst index (1–24 h ahead), each with a value, a storm probability and an alert. Forecasts are
   issued every hour and verified against observations every day.
3. **Distribution.** A public API, Parquet/JSON files and a website.

An experimental aurora-visibility forecast runs in the backend (see "Aurora visibility"). It
is not yet shown on the website.

The Dst index complements Kp because it better represents low latitudes.

### Architecture

No server or database runs continuously. All outputs are precomputed and stored as files.

```
 GitHub Actions (hourly, triggered externally; see "Scheduling")
   ├─ futurespace update     connectors download and standardize each source;
   │                         the writer merges, deduplicates and writes Parquet + JSON
   ├─ futurespace predict    applies the published models to the latest data
   ├─ futurespace aurora     aurora-visibility forecast per city
   └─ futurespace verify     (daily) compares past forecasts with observations
                    │
                    ▼
 Hugging Face dataset (data and models, versioned by commit)
                    │
                    ▼
 FastAPI on Vercel (reads JSON only, with caching)  ──▶  API clients / website

 GitHub Actions (monthly): futurespace train --publish-if-better
```

- **Idempotent writes:** repeating a collection creates no duplicates. Values revised by the
  source (e.g. preliminary → definitive Kp) replace earlier ones.
- **Fault isolation:** an unavailable source does not stop the others; the failure is logged.
- **Partitioning:** by day (1–5 min data), month (OMNI, flares, regions, forecasts) or year
  (Kp, Dst, CMEs). Each dataset has a `manifest.json` listing its partitions and coverage.

## Data sources

All sources are public and free of charge, and none requires registration.

| Dataset | Stage | Provider | Cadence | Coverage | Use |
|---|---|---|---|---|---|
| `omni-hourly` | all | NASA GSFC OMNI2 (CDAWeb HAPI) | 1 h | 1963– | training (solar wind and indices) |
| `kp-gfz` | ground | GFZ Potsdam | 3 h | 1932– | training labels (definitive Kp) |
| `cme-cactus` | Sun | SIDC CACTus (SOHO/LASCO) | event | 1997– | CME features |
| `kp-forecast-3day` | ground | NOAA SWPC 3-Day Forecast (NCEI archive) | 12 h | 2022– | benchmark |
| `solar-wind-plasma` | L1 | NOAA SWPC RTSW (SOLAR-1, IMAP, ACE) | 1 min | since collection | live inputs |
| `solar-wind-mag` | L1 | NOAA SWPC RTSW | 1 min | since collection | live inputs |
| `kp` | ground | NOAA SWPC | 3 h | since collection | live inputs, verification |
| `kp-forecast` | ground | NOAA SWPC | 3 h | since collection | live benchmark |
| `dst` | ground | WDC Kyoto (via NOAA SWPC) | 1 h | since collection | live inputs, verification |
| `goes-xray` | orbit | NOAA SWPC / GOES | 1 min | since collection | archived |
| `goes-protons` | orbit | NOAA SWPC / GOES | 5 min | since collection | archived |
| `goes-flares` | Sun | NOAA SWPC / GOES | event | since collection | archived |
| `solar-regions` | Sun | NOAA SWPC | 1 day | since collection | archived |
| `aurora-ovation` | all | NOAA SWPC OVATION Prime | 1 h | since collection | aurora probability map (Parquet only) |
| `aurora-weather` | ground | Open-Meteo | 1 h | since collection | cloud cover at the aurora sites |
| `predictions` | — | this project | 1 h | since model release | live forecasts |
| `verification` | — | this project | 1 h | since model release | forecast vs. observed |
| `aurora-visibility` | — | this project | 1 h | since collection | aurora-visibility forecast (Parquet only) |

<details>
<summary>Source details (endpoints, conventions, caveats)</summary>

- **NASA OMNI2** (`OMNI2_H0_MRG1HR`, https://cdaweb.gsfc.nasa.gov/hapi): solar wind from
  multiple spacecraft since 1963 (IMP, ISEE, Wind, ACE, DSCOVR), intercalibrated and
  time-shifted to the bow shock, with Kp, Dst, AE and F10.7 on the same time axis. Released
  with a delay of a few weeks. Applied conventions: `time` is the start of the hour (HAPI
  timestamps HH:30), Kp is divided by 10, and fill values (`9999.9`, etc.) become missing.
- **GFZ Kp** (`https://kp.gfz.de/app/json/`): definitive Kp since 1932, with a `status` field
  (`def`, `pre`, `now`). CC BY 4.0.
- **CACTus:** automatic CME detection on LASCO C2/C3 images. The connector merges three
  products of the same algorithm:
  - the science catalog (1997–2015);
  - the monthly quicklook catalog (2015–);
  - the real-time feed (last week).

  Real-time detections are discarded once the quicklook catalog covers their period, because
  the same event appears in both with slightly different values. NASA DONKI was unavailable
  during development and only starts in 2010; it is planned as a complementary source.
- **NOAA RTSW** (`/json/rtsw/rtsw_{wind,mag}_1m.json`): several spacecraft are reported at
  once. `source` identifies the spacecraft and `active` marks NOAA's operational one, which is
  the one used for forecasting. The source retains only about 24 h of data.
- **NOAA Kp forecast** (`noaa-planetary-k-index-forecast.json`): the source shows only the
  current forecast. Every issued version is archived (`time` = collection time,
  `target_time` = forecast block), including the `estimated` future blocks.
- **NOAA 3-Day Forecast archive** (NCEI, `…/swpc-products/daily_reports/3day_forecast/`): all
  editions (00:30 and 12:30 UTC) since March 2022. Only blocks that had not started at issue
  time are kept. September and October 2022 are missing from the archive.
- **Kyoto Dst** (`kyoto-dst.json`): provisional real-time values.
- **OVATION Prime** (`ovation_aurora_latest.json`): probability of visible aurora on a global
  1° grid, computed by NOAA from the L1 solar wind and valid for roughly 30–90 minutes ahead.
  Only non-zero cells are stored (~20,000 per map). NOAA keeps no public archive of this
  product, so the history starts with this collection. Longitudes follow the source
  convention (0–359).
- **Open-Meteo** (`api.open-meteo.com/v1/forecast`): hourly total, low, mid and high cloud
  cover and horizontal visibility for the next 25 hours at each aurora site, in a single
  request. Each hour keeps the most recent forecast; `issued` records when it was obtained.
  CC BY 4.0, free for non-commercial use.
- **GOES and solar regions:**
  - X-ray flux in two bands;
  - integral proton flux (≥1 to ≥500 MeV);
  - the flare list;
  - active regions with McIntosh and Mount Wilson classes and NOAA's C/M/X flare probabilities.

</details>

NOAA real-time products are in the public domain. The `license` field of each dataset in the
API states how to credit the original source.

## Forecasting models

### Targets

| Index | Horizons | Storm definition |
|---|---|---|
| Kp | 3, 6, 12, 24 h | Kp ≥ 5− (4.67), NOAA G1 or stronger |
| Dst | 1, 3, 6, 12, 24 h | Dst ≤ −50 nT |

Each index and horizon has two models:

- a **regressor**, which forecasts the value at *t + h*;
- a **classifier**, which gives the probability of a storm at *t + h*. An alert is raised when
  the probability exceeds a threshold selected on the validation set.

Kp targets refer to the 3-hour block that contains *t + h*.

### Features

Only variables that are also available in real time are used. This allows a model trained on
OMNI to run on the live data. Feature version 3 has 82 variables in five groups:

| Group | Variables | Rationale |
|---|---|---|
| Solar wind at L1 (55) | Bz, By, \|B\|, speed, density, temperature and dynamic pressure; Newell coupling `v^(4/3) B_T^(2/3) sin^(8/3)(θ/2)`; VBs = v·max(−Bz, 0); clock angle (sin, cos); for Bz, v, n, Newell and VBs, lags of 1–6 h and means over 3–24 h; minimum Bz over 3–12 h | energy entering the magnetosphere and its trend |
| Current geomagnetic state (11) | Dst, its lags, its 1 h and 3 h differences and its 24 h minimum; last completed Kp block, the previous block and the 24 h maximum | an ongoing storm |
| 27-day recurrence (4) | solar wind speed one solar rotation earlier; speed, Kp and Dst over the corresponding next 24 h, one rotation earlier | recurrent high-speed streams |
| CMEs (8) | fastest wide CME (width ≥ 90°) in the last 24/48/72/96 h; number of wide CMEs in 72 h; ballistic arrival estimate (time to arrival, speed and width of the nearest CME) | lead time of hours to days |
| Calendar (4) | day of year and UT hour (sin, cos) | seasonal (Russell–McPherron) and diurnal effects |

Leakage controls, enforced by automated tests:

- the row for hour *t* uses only data up to the end of hour *t*;
- a Kp value is used only after its 3-hour block has ended;
- a CME becomes available to the model 6 h after its onset, the typical latency of the
  real-time catalog;
- before the start of the CME catalog, CME features are missing rather than zero.

### Model and hyperparameters

The models are scikit-learn `HistGradientBoostingRegressor` and
`HistGradientBoostingClassifier` (version pinned at 1.9.1). This algorithm was chosen because
it:

- performs well on tabular data;
- trains in minutes on a CPU;
- handles missing values natively;
- can be inspected.

| Hyperparameter | Value |
|---|---|
| learning rate | 0.05 |
| max leaf nodes | 31 |
| min samples per leaf | 40 |
| L2 regularization | 1.0 |
| early stopping | disabled |
| random state | 0 |
| regressor iterations | {200, 500, 1000}, selected by validation RMSE |
| classifier iterations | 400 |
| alert threshold | 0.02–0.90 (step 0.01), selected by validation HSS |

The published version, `v3-20261106`, contains 9 regressor/classifier pairs, with 8,400 trees
and 512,400 nodes in total.

### Training and evaluation

1. **Temporal split:**
   - training: 1995-01-01 to 2014-12-31;
   - validation: 2015-01-01 to 2018-12-31;
   - test: 2019-01-01 onward.

   The split is never random.
2. **Iterations:** regressors are fitted on the training set for each iteration count, and
   the one with the lowest validation RMSE is kept.
3. **Alert threshold:** the classifier is fitted on the training set, and the threshold that
   maximizes HSS on the validation set is selected. Maximizing TSS instead produced 60–95%
   false alarms, because TSS barely penalizes false alarms for rare events.
4. **Evaluation:** the models are refitted on training + validation and evaluated once on the
   test set. These are the metrics reported below.
5. **Published model:** the model is refitted on all data, including the test period, before
   publication. The reported metrics remain those of step 4.

**Baselines:**

- *persistence:* the current Dst, or the last completed Kp block;
- *climatology:* the storm frequency in the training data, used as the reference for BSS.

**Metrics**, with hits *a*, false alarms *b*, misses *c* and correct negatives *d*:

| Metric | Definition |
|---|---|
| RMSE, MAE | √mean(ŷ − y)², mean\|ŷ − y\| |
| POD | a / (a + c) |
| FAR | b / (a + b) |
| TSS | POD − b / (b + d) |
| HSS | 2(ad − bc) / [(a + c)(c + d) + (a + b)(b + d)] |
| Brier score | mean(p − o)² |
| BSS | 1 − Brier / Brier of climatology |

**Retraining:** the model is retrained monthly in GitHub Actions. A new version is published
only if, on the same test split:

- its mean RMSE is at most 1.02 times that of the current version; and
- its mean BSS has not dropped by more than 0.02.

The models are not updated online from their errors.

### Results

The figures below are for version `v3-20261106` on the test period 2019-01-01 to 2026-10-30:
68,600 hours, of which 2,174 were Dst storm hours and 1,584 were Kp storm hours.

| Forecast | RMSE | RMSE persistence | Reduction | POD | FAR | TSS | HSS | HSS persistence | BSS | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|
| Dst +1 h | 3.82 | 4.40 | 13% | 0.93 | 0.12 | 0.92 | 0.90 | 0.87 | 0.85 | 0.42 |
| Dst +3 h | 7.48 | 8.88 | 16% | 0.83 | 0.29 | 0.82 | 0.76 | 0.73 | 0.67 | 0.28 |
| Dst +6 h | 10.53 | 12.58 | 16% | 0.72 | 0.37 | 0.70 | 0.66 | 0.61 | 0.53 | 0.27 |
| Dst +12 h | 14.08 | 16.79 | 16% | 0.50 | 0.50 | 0.48 | 0.48 | 0.45 | 0.33 | 0.24 |
| Dst +24 h | 16.76 | 20.59 | 19% | 0.41 | 0.73 | 0.37 | 0.30 | 0.28 | 0.16 | 0.12 |
| Kp +3 h | 0.76 | 1.01 | 25% | 0.60 | 0.54 | 0.59 | 0.51 | 0.40 | 0.36 | 0.21 |
| Kp +6 h | 0.91 | 1.17 | 23% | 0.58 | 0.72 | 0.54 | 0.36 | 0.29 | 0.22 | 0.12 |
| Kp +12 h | 1.04 | 1.34 | 22% | 0.41 | 0.83 | 0.36 | 0.21 | 0.19 | 0.08 | 0.09 |
| Kp +24 h | 1.15 | 1.50 | 23% | 0.31 | 0.90 | 0.24 | 0.12 | 0.10 | 0.04 | 0.06 |

RMSE is in nT for Dst and in Kp units for Kp. POD, FAR, TSS and HSS refer to the alert.

- **Value:** the forecast value outperforms persistence at every horizon.
- **Alerts:** the alert is skillful up to about 6 h. At 12–24 h, most Kp alerts are false.
- **CME features:** the previous feature set without CME variables (v2, 70 features) gives BSS
  within ±0.02 at every horizon. The CACTus catalog does not indicate whether a CME is
  Earth-directed.

**Feature-group importance.** The variables of each group are permuted jointly on the test
period (20,000 sampled hours, 3 repetitions), and the increase in error is measured: RMSE for
the regressor, Brier score for the classifier. Each value is the group's share of the total
increase. These values were measured with the v3 code on data through 2026-10-30, in a local run;
training records stored from the next monthly run onward will include them (`/v1/models`).

| Model | Solar wind | Geomagnetic state | Recurrence | CMEs | Calendar |
|---|---|---|---|---|---|
| Kp +3 h, value | 78% | 17% | 0% | 0% | 5% |
| Kp +3 h, probability | 77% | 16% | 0% | 1% | 6% |
| Kp +24 h, value | 54% | 12% | 19% | 8% | 8% |
| Kp +24 h, probability | 40% | 10% | 18% | 15% | 17% |
| Dst +6 h, value | 23% | 75% | 0% | 0% | 1% |
| Dst +6 h, probability | 22% | 76% | 0% | 0% | 2% |

Short-range Kp depends mainly on the incoming solar wind. Permuting that group raises the
Kp +3 h RMSE from 0.76 to 1.34. Short-range Dst depends mainly on its current value and trend.
At 24 h, recurrence and CMEs account for 27% of the Kp value model. From the next training run
onward, this breakdown is stored for all nine models (see "Model history").

**Comparison with NOAA (2022–2026).** Reproducible with `futurespace backfill kp-forecast-3day
--start 2022-03-01`, `futurespace train --no-refit --out models-2018` and `futurespace benchmark
--bundle models-2018/bundle.joblib`. The comparison uses the NCEI archive of NOAA's 3-Day
Forecast. Both forecasts are scored against observed Kp (GFZ), on the same blocks and with
the same lead times. For each NOAA edition, the model:

- uses only data available up to the previous full hour;
- is the version trained up to 2018, which never saw this period.

| Lead time | Blocks | MAE model | MAE NOAA | Alert HSS model | Alert HSS NOAA |
|---|---|---|---|---|---|
| ~6 h | 2,528 | 0.72 | 1.16 | 0.40 | 0.24 |
| ~12 h | 2,528 | 0.80 | 1.18 | 0.21 | 0.16 |
| ~24 h | 2,528 | 0.90 | 1.16 | 0.08 | 0.16 |

The model has lower error at all three lead times and better alerts at 6 h and 12 h. NOAA has
better alerts at 24 h. About 82% of NOAA's G1 alerts in this period were not confirmed.

This hindcast uses OMNI and definitive Kp, which are cleaner than real-time data. The live
verification described below tests whether the result holds in operation.

**Live verification.** `futurespace verify` runs daily. It pairs each issued forecast with
the value observed later and records the error, the persistence error and, for Kp, the error
of the NOAA forecast issued up to the same moment. Seven- and 30-day summaries are available
at `GET /v1/verification/summary`.

### Limitations

- **Storm onset.** The model mainly sees solar wind measured at L1, about 1 h before it reaches
  Earth. The CME features do not include direction.
- **Extreme events.** Tree ensembles do not extrapolate beyond the training range, so events
  such as May 2024 (Dst −406 nT) are underestimated.
- **Training versus operation.**
  - OMNI is time-shifted to the bow shock, whereas live data are raw L1 measurements taken
    about 45 minutes earlier.
  - Provisional Dst and changes of the L1 spacecraft can introduce discontinuities that are
    absent from OMNI.
- **Interpretation of importance.** Permutation importance describes what the model uses, not
  physical causation. Correlated groups share credit.

## Aurora visibility (experimental)

This component runs in the backend to accumulate data and forecasts. It is not yet shown on
the website.

Every hour, `futurespace aurora` computes, for 37 sites and for each of the next 24 hours, a
score from 0 to 1 for the conditions to see the aurora (higher means better conditions). The sites are listed in `src/futurespace/aurora/sites.py`
and include:

- cities in and near the auroral zones;
- mid-latitude cities;
- Southern Hemisphere and Brazilian references;
- locations of NASA THEMIS all-sky cameras.

The score is not a probability: its weights have not yet been compared with observed
aurorae, so it can rank sites and hours but does not state how often the aurora will appear.
A calibrated probability will be added as a separate column once observations are available.

The score is computed as:

```
visibility_score = aurora_score × (1 − cloud_cover/100) × darkness × moon_factor
```

| Component | Method (v0) |
|---|---|
| expected Kp | the project's Kp forecast for the 3-hour block, interpolated between the +3/+6/+12/+24 h blocks; before the first forecast block, the last measured Kp |
| `aurora_score` | logistic function of (expected Kp − Kp needed), with a scale of 0.5 Kp; the Kp needed comes from the visibility line used by the website's aurora map, 66° − 1.5° × Kp of dipole magnetic latitude |
| `darkness` | 0 with the Sun above −6°, 1 below −15°, linear in between |
| `moon_factor` | 1 − 0.3 × illuminated fraction when the Moon is above the horizon |
| cloud cover | Open-Meteo forecast |

**OVATION reference.** For the next hour, the maximum OVATION probability within 500 km is
stored separately, together with the corresponding visibility (`visibility_score_ovation`).
At an altitude of about 110 km, the aurora appears on the horizon from that distance.

The two estimates are kept apart because they are on different scales: OVATION rarely
exceeds about 50%, and it depends on magnetic local time, which the Kp rule ignores. The
OVATION archive collected here is intended to calibrate the next version of the formula.

**Astronomy.** Sun and Moon positions use the Astronomical Almanac low-precision formulas,
implemented in `aurora/astro.py`. Against astropy, over 300 random times and places between
1990 and 2035, the maximum errors are:

- 0.008° for the Sun's altitude;
- 0.3° for the Moon's altitude;
- 0.002 for the Moon's illuminated fraction.

**Limitations.** The v0 weights are not calibrated against observations. The Kp rule has no
magnetic-local-time dependence and no poleward boundary for the auroral oval.

**Planned verification:** classify all-sky camera images (THEMIS, with labelled data such as
OATH) and score both estimates against the observed aurora.

## Model history and reproducibility

Each training run writes a record (`report.json`) next to its models. `models/index.json`
lists every run, including those rejected by the publication gate, together with the reason
for rejection.

| Field | Content |
|---|---|
| `reproducibility.code` | GitHub commit used for training; whether there were uncommitted changes |
| `reproducibility.data` | Hugging Face dataset commit that was read; rows and coverage per source |
| `reproducibility.hyperparameters` | tree parameters, search grids, horizons, storm definition, importance settings |
| `reproducibility.environment` | Python, scikit-learn, numpy and pandas versions |
| `splits`, `features`, `feature_groups` | temporal split, ordered feature list, group membership |
| `models[]` | per model: iterations, threshold, row counts, test metrics, feature-group importance |

```bash
curl https://future-space-six.vercel.app/v1/models                # all runs
curl https://future-space-six.vercel.app/v1/models/v3-20261106    # one run
```

Training is deterministic (`random_state=0`, no early stopping). A model can therefore be
reproduced by:

1. checking out the recorded commit;
2. installing the versions pinned in `requirements-pipeline.txt`;
3. reading the dataset at the recorded revision (`HuggingFaceStorage(repo, revision=<sha>)`).

Version `v3-20261106` predates these records, so its code and data revision fields are
empty. Its training data end at 2026-10-30 14:00 UTC, and it was published in dataset commit
`20c2da27`.

## Data access

### Files

The Parquet files can be read directly, without the API:

```python
import pandas as pd  # requires pandas, pyarrow and huggingface_hub

omni = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/omni-hourly/")
kp = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/kp-gfz/")
cmes = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/cme-cactus/")
```

```
parquet/<dataset>/<partition>.parquet         analysis and training
json/<dataset>/<partition>.json               read by the API
json/<dataset>/manifest.json                  partitions and coverage
json/<dataset>/latest.json                    latest values
json/catalog.json                             catalog
models/current.json                           model version in use
models/index.json                             history of all training runs
models/<version>/bundle.joblib                trained models
models/<version>/report.json                  record of that version
models/rejected/<version>-<time>/report.json  runs rejected by the publication gate
```

| Dataset | Parquet | CSV | JSON |
|---|---|---|---|
| OMNI, 1963–present (560k rows) | 25 MB | 60 MB | 148 MB |
| GFZ Kp, 1932–present (277k rows) | 2.6 MB | — | 15 MB |
| L1 magnetic field, 1 day, 3 spacecraft | ~50 KB | — | ~340 KB |

### API

Base URL: `https://future-space-six.vercel.app`. Interactive documentation is at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /v1/datasets` | datasets and their coverage |
| `GET /v1/datasets/{id}` | metadata, fields, units, license and coverage |
| `GET /v1/datasets/{id}/latest` | latest values |
| `GET /v1/datasets/{id}/data` | time series for a range (`start`, `end`, `fields`, `format=json\|csv`) |
| `GET /v1/datasets/predictions/latest` | latest forecasts |
| `GET /v1/compare/kp?date=YYYY-MM-DD` | model, NOAA and observed Kp by 3-hour block |
| `GET /v1/compare/dst?date=YYYY-MM-DD` | model forecasts and observed Dst by hour |
| `GET /v1/verification/summary` | forecast errors over the last 7 and 30 days |
| `GET /v1/models` | model history, metrics and feature-group importance |
| `GET /v1/models/{id}` | record of a training run |
| `GET /v1/datasets/aurora-visibility/latest` | latest aurora-visibility forecast (all sites, 0–24 h) |

```bash
curl "https://future-space-six.vercel.app/v1/datasets/omni-hourly/data?start=2024-05-10&end=2024-05-12&fields=bz_gsm,speed,kp,dst"
```

Large datasets (`aurora-ovation`, `aurora-visibility`) are available only as Parquet files.
For them, `/data` returns `404` with a link to the files.

Usage limits:

- each dataset has a maximum range per request (`max_query_days`; for example, 3 days for
  1-minute data); large volumes should be read from the Parquet files;
- 60 requests per minute per IP; beyond that the API returns `429` with `Retry-After`;
- unknown query parameters are rejected with `400`.

See [SECURITY.md](SECURITY.md).

## Running the project

Requires Python 3.11 or later.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,hf]"

futurespace datasets                                 # list the catalog
futurespace update                                   # collect real-time data into ./data
futurespace update --schedule daily                  # recent OMNI and GFZ Kp
futurespace backfill omni-hourly --start 1963-01-01  # full OMNI history (~4 min)
futurespace backfill kp-gfz --start 1932-01-01       # full Kp history (~30 s)
futurespace backfill cme-cactus --start 1997-01-01   # CMEs since 1997 (~40 s)
futurespace train                                    # train and evaluate (~10 min) into ./models
futurespace train --no-cme --out models-no-cme       # same training without CME features
futurespace predict                                  # forecast from current data
futurespace aurora                                   # aurora-visibility forecast per city
futurespace serve --reload                           # API at http://localhost:8000/docs
pytest -q                                            # tests
```

| Variable | Purpose | Default |
|---|---|---|
| `FS_STORAGE` | pipeline storage backend: `local`, `hf` or `s3` | `local` |
| `FS_DATA_DIR` | directory for `local` | `data` |
| `FS_HF_REPO` | Hugging Face dataset, e.g. `Strutf/future-space` | — |
| `HF_TOKEN` | Hugging Face write token (pipeline only) | — |
| `FS_DATA_URL` | alternative base URL from which the API reads JSON | — |
| `FS_RATE_LIMIT_PER_MINUTE` | API requests per minute per IP (`0` disables the limit) | `60` |
| `FS_S3_BUCKET`, `FS_S3_ENDPOINT_URL` | settings for `s3` (Cloudflare R2, Backblaze B2, etc.) | — |

The API reads from `FS_DATA_URL` if it is set, otherwise from `FS_HF_REPO`, otherwise from
the local directory.

### Deployment

The project runs entirely on free tiers:

1. **Hugging Face:** a public dataset repository, written with a fine-grained token scoped to
   that repository.
2. **GitHub:** a public repository, where Actions minutes are unlimited. Configuration:
   - secret `HF_TOKEN`, the token from step 1;
   - variable `FS_HF_REPO`.

   The token exists only in GitHub secrets.
3. **Initial load:** run the backfills with `FS_STORAGE=hf`, or upload a local `data/`
   directory with `hf upload <repo> data . --repo-type dataset`.
4. **Models:** run the `train` workflow from the Actions tab.
5. **Vercel:** import the repository and set `FS_HF_REPO`. The API reads only public data and
   needs no token.

Security recommendations are in [SECURITY.md](SECURITY.md).

### Scheduling

GitHub's native `on.schedule` trigger never fired in this repository, even with a 5-minute
interval, although manual runs work. Workflows are therefore triggered by
[cron-job.org](https://cron-job.org) through GitHub's `workflow_dispatch` API:

| Job | Schedule (UTC−3) | Workflow | Request body |
|---|---|---|---|
| ingest, real-time | `23 * * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"realtime"}}` |
| ingest, daily | `47 0 * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"daily"}}` |
| training | `41 1 1 * *` | `train.yml` | `{"ref":"main"}` |

Each job configuration:

- **method:** `POST`;
- **headers:** `Authorization: Bearer <token>`, `Accept: application/vnd.github+json` and
  `Content-Type: application/json`;
- **token:** fine-grained, with only *Actions: read and write* on this repository;
- **expected response:** `204`.

The native schedule remains in the workflows as a fallback. All writing jobs share a
concurrency group, so they never write at the same time.

Historical loads in production: run the `ingest` workflow manually with `backfill` (dataset
id) and `desde` (start date).

Free-tier constraints:

- GitHub disables scheduled workflows after 60 days without repository activity.
- L1 solar wind is retained at the source for only about 24 h, so more than one day without
  collection loses data.
- Vercel Hobby limits responses to about 4.5 MB, which is why the API limits the range per
  request.

## Website

https://future-space-six.vercel.app is rendered by the API itself, without a build step, in
Portuguese, English, Spanish and French.

| Page | Path (EN) | Content |
|---|---|---|
| Home | `/en` | project summary and current forecast |
| Forecasts | `/en/dashboard` | three tabs for yesterday, today and tomorrow: Kp (model, NOAA, observed; default), model storm probability, hourly Dst |
| About | `/en/about` | storm physics, Kp, Dst, CMEs and the formulas used |
| Interactive | `/en/interactive` | CME simulator, historical storm replays, aurora map |

The interactive pages run in the browser from static files generated by
`futurespace export-web --bundle <models trained up to 2018>`:

- `sim-model.json`: a reduced model with monotonic constraints (6-hour means of speed,
  density and VBs). It is exported from scikit-learn to JSON and evaluated in JavaScript; a
  test verifies that both give identical outputs.
- `replays.json`: hourly data for the 1989, 2003, 2015 and 2024 storms, with the forecasts
  issued 3 h earlier.
- `world-land.json`: continent outlines (Natural Earth, public domain).

The aurora map uses a dipole approximation of magnetic latitude and a visibility boundary
calibrated on NOAA aurora maps: about 66° − 1.5° × Kp of magnetic latitude.

Page templates are in `src/futurespace/api/web/pages`, and texts are in
`web/locales/{pt,en,es,fr}.json`. Tests ensure that all locale files have the same keys and
placeholders.

## Repository structure

```
src/futurespace/
├── catalog.py          dataset metadata (fields, units, licenses, schedules)
├── partitions.py       file layout and time utilities
├── storage.py          storage backends: local, Hugging Face, S3, HTTP
├── model_index.py      model history (models/index.json), readable without ML libraries
├── cli.py              `futurespace` command
├── aurora/
│   ├── sites.py        sites followed by the aurora forecast
│   ├── astro.py        Sun and Moon altitude, Moon illumination
│   └── visibility.py   aurora-visibility forecast (v0)
├── pipeline/           HTTP client, writer (merge and deduplication), one connector per source
├── ml/
│   ├── features.py     hourly table, features, targets and feature groups
│   ├── train.py        training, temporal validation, evaluation, importance, run records
│   ├── metrics.py      RMSE, MAE, POD, FAR, TSS, HSS, Brier score
│   ├── serve.py        live forecasts and model publication
│   ├── verify.py       forecast verification
│   └── export_web.py   browser models and storm replays
└── api/
    ├── app.py          FastAPI application (no pandas dependency, for serverless use)
    ├── compare.py      model and NOAA comparison used by the dashboard
    └── web/            page templates, static files, translations
api/index.py, vercel.json    Vercel entry point
.github/workflows/           ingest, train, tests
requirements-pipeline.txt    pinned versions for the workflows that hold the token
```

To add a source:

1. describe the dataset in `catalog.py`;
2. implement `(client, start, end) -> DataFrame` in `pipeline/connectors/`;
3. register it in `connectors/__init__.py`.

The writer and the API then serve it without further changes.

## Roadmap

- [x] Collection and archiving of 15 datasets; public API; Parquet/JSON dataset
- [x] Kp and Dst forecasts 1–24 h ahead (value, probability, alert)
- [x] CME features from CACTus
- [x] Hourly live forecasts, daily verification, comparison with NOAA
- [x] Monthly retraining with a publication gate
- [x] Reproducible training records and feature-group importance
- [x] Website in four languages with interactive pages
- [x] Aurora-visibility forecast per city in the backend (v0), with the OVATION archive
- [ ] Aurora: calibration with OVATION and all-sky cameras; publication on the website
- [ ] CME direction and arrival time (NASA DONKI, WSA–Enlil)
- [ ] Additional sources: INTERMAGNET magnetometers (Vassouras and Tatuoca, Brazil),
      INPE/EMBRACE, ionospheric TEC
- [ ] Flare forecasting (SDO/HMI SHARP parameters); linking events across sources;
      anomaly detection
- [ ] Probability calibration analysis and storm-onset metrics

## Citation and data credits

Citation metadata for this repository is in [`CITATION.cff`](CITATION.cff). To cite a
specific model, include its version and record URL, e.g.
`https://future-space-six.vercel.app/v1/models/v3-20261106`.

Original data sources:

- **OMNI:** King, J. H., & Papitashvili, N. E. (2005). *JGR*, 110, A02104.
  doi:10.1029/2004JA010649. NASA GSFC SPDF / OMNIWeb.
- **Kp:** Matzka, J., et al. (2021). *Space Weather*, 19, e2020SW002641.
  doi:10.1029/2020SW002641. Data: doi:10.5880/Kp.0001 (CC BY 4.0).
- **Dst:** World Data Center for Geomagnetism, Kyoto, et al. (2015). doi:10.17593/14515-74000.
- **CACTus:** Robbrecht, E., & Berghmans, D. (2004). *A&A*, 425, 1097–1106.
  doi:10.1051/0004-6361:20041302. SIDC, Royal Observatory of Belgium.
- **NOAA SWPC** real-time products and the NCEI forecast archive: public domain.
- **OVATION Prime:** Newell, P. T., Sotirelis, T., & Wing, S. (2009). *JGR*, 114, A09207.
  doi:10.1029/2009JA014326.
- **Open-Meteo:** weather data by Open-Meteo.com (CC BY 4.0).

Methods referenced:

- Newell, P. T., et al. (2007). *JGR*, 112, A01206. doi:10.1029/2006JA012015.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

## License

The code is released under the MIT License. Each dataset retains the license of its original
source, given in the catalog (`/v1/datasets/{id}`).
