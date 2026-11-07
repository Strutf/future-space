# Future Space

**Open multi-source space-weather data and machine-learning forecasts of geomagnetic storms**

> 🇧🇷 Versão em português: [README.pt.md](README.pt.md)

| | |
|---|---|
| **Website** (PT · EN · ES · FR) | https://future-space-six.vercel.app |
| **API** (OpenAPI docs) | https://future-space-six.vercel.app/docs |
| **Data and models** (Parquet, JSON, joblib) | https://huggingface.co/datasets/Strutf/future-space |
| **Model history** | https://future-space-six.vercel.app/v1/models |
| **Code** | https://github.com/Strutf/future-space (MIT) |

## Abstract

Future Space collects 13 public space-weather datasets from six providers (NOAA SWPC,
NASA OMNI, GFZ Potsdam, WDC Kyoto, SIDC/CACTus and the NCEI forecast archive). Every hour it
harmonizes them into a single time-indexed format and publishes them as an open dataset
and a free API. On top of this dataset we train gradient-boosted decision trees that forecast
the **Kp** index (3–24 h ahead) and the **Dst** index (1–24 h ahead). Each forecast includes
a value, a storm probability and a calibrated alert.

The models use only inputs that are also available in real time: solar wind at L1,
current geomagnetic state, 27-day recurrence, coronal mass ejections (CMEs) and calendar
terms. They are trained on 1995–2014, tuned on 2015–2018 and evaluated once on
2019–2026, a period that includes the May 2024 superstorm. On this test period the models
reduce RMSE by 13–25% relative to persistence at every horizon. The storm alert is skillful
up to about 6 h (Dst +3 h: POD 0.83, FAR 0.29, HSS 0.76).

In a like-for-like comparison with NOAA's official 3-day Kp forecast (2022–2026, 2,534
forecast blocks), the model has lower mean absolute error at 6, 12 and 24 h lead times. It
also has a higher alert HSS at 6 and 12 h, while NOAA is better at 24 h.

Permutation importance shows that short-range Kp skill comes mostly from the L1 solar wind
(78% at +3 h). Short-range Dst skill comes mostly from the current geomagnetic state
(75% at +6 h). At +24 h, recurrence and CMEs contribute about 27%. Predictions are issued
hourly and verified daily against observations. Every trained model is recorded with the
code commit, data revision and hyperparameters needed to reproduce it.

> ⚠️ The forecasts are **experimental**. Do not use them for operational decisions; use
> [NOAA SWPC](https://www.swpc.noaa.gov/) for that.

---

## Contents

1. [Motivation](#1-motivation)
2. [Data](#2-data)
3. [Data pipeline](#3-data-pipeline)
4. [Methods](#4-methods)
5. [Evaluation protocol](#5-evaluation-protocol)
6. [Results](#6-results)
7. [Limitations](#7-limitations)
8. [Reproducibility](#8-reproducibility)
9. [Using the data and the API](#9-using-the-data-and-the-api)
10. [Operations](#10-operations)
11. [Website](#11-website)
12. [Code structure](#12-code-structure)
13. [Roadmap](#13-roadmap)
14. [How to cite](#14-how-to-cite)
15. [License and data credits](#15-license-and-data-credits)

---

## 1. Motivation

A solar storm is a single chain of events, observed by different instruments at each stage:

```
   SUN                  SPACE (L1)                EARTH ORBIT              GROUND
 flares, active   ──▶  solar wind and        ──▶  X-rays, energetic   ──▶  Kp and Dst indices,
 regions, coronal      magnetic field             protons                  magnetometers,
 mass ejections        (~1 h before Earth)                                 ionosphere / GPS
```

The forecasting signal lives in the **relationships between these stages**. Existing
sources are scattered across agencies, formats and time conventions, and much of the
real-time data disappears after 1–7 days. The project therefore has three goals:

1. **Data:** archive and harmonize the sources continuously, so they can be studied together
   and so real-time-only data (e.g. L1 solar wind, kept for ~24 h at the source) is not lost.
2. **Models:** forecast geomagnetic activity using only information that is available in
   real time. Evaluate the forecasts honestly against simple baselines and against the
   official NOAA forecast.
3. **Openness:** make data, models, predictions and evaluation freely available, at zero
   infrastructure cost.

The Dst index is included alongside Kp because it better represents low latitudes, such as
Brazil, where the project originated.

## 2. Data

All sources are **public and free**, and none requires registration. Every table has a
`time` column in UTC.

| Dataset | Stage | Provider | Cadence | Coverage | Role |
|---|---|---|---|---|---|
| `omni-hourly` | all | NASA GSFC OMNI2 (via CDAWeb HAPI) | 1 h | **1963–** | main training set (solar wind + indices) |
| `kp-gfz` | ground | GFZ Potsdam | 3 h | **1932–** | official Kp, training labels |
| `cme-cactus` | Sun | SIDC CACTus (SOHO/LASCO) | event | **1997–** | CME features |
| `kp-forecast-3day` | ground | NOAA SWPC 3-Day Forecast (NCEI archive) | 12 h | **2022–** | benchmark |
| `solar-wind-plasma` | L1 | NOAA SWPC RTSW (SOLAR-1, IMAP, ACE) | 1 min | since collection | live inputs |
| `solar-wind-mag` | L1 | NOAA SWPC RTSW | 1 min | since collection | live inputs |
| `kp` | ground | NOAA SWPC | 3 h | since collection | live inputs, verification |
| `kp-forecast` | ground | NOAA SWPC | 3 h | since collection | live benchmark (every issued forecast) |
| `dst` | ground | WDC Kyoto (via NOAA SWPC) | 1 h | since collection | live inputs, verification |
| `goes-xray` | orbit | NOAA SWPC / GOES | 1 min | since collection | archived (future flare models) |
| `goes-protons` | orbit | NOAA SWPC / GOES | 5 min | since collection | archived (radiation storms) |
| `goes-flares` | Sun | NOAA SWPC / GOES | event | since collection | archived |
| `solar-regions` | Sun | NOAA SWPC | 1 day | since collection | archived (official flare probabilities) |
| `predictions` | — | this project | 1 h | since model release | live forecasts |
| `verification` | — | this project | 1 h | since model release | forecast vs. observed |

<details>
<summary><b>Source details</b> (endpoints, conventions, caveats)</summary>

- **NASA OMNI2** (`OMNI2_H0_MRG1HR`, https://cdaweb.gsfc.nasa.gov/hapi): solar wind from
  dozens of spacecraft since 1963 (IMP, ISEE, Wind, ACE, DSCOVR…), intercalibrated and
  time-shifted to the bow shock, with Kp, Dst, AE and F10.7 on the same time axis.
  Published with a few weeks of delay. Conventions: `time` is the start of the hour (HAPI
  stamps HH:30), Kp is divided by 10 and fill values (`9999.9`…) become missing.
- **GFZ Kp** (`https://kp.gfz.de/app/json/`): definitive Kp since 1932, with `status`
  (`def`, `pre`, `now`). CC BY 4.0.
- **CACTus**: automatic CME detection on LASCO C2/C3 images. The connector merges the science
  catalog (1997–2015), the monthly quicklook (2015–) and the real-time feed (last week). When
  the quicklook covers a period, the real-time detections for that period are discarded,
  because the same event appears in both with slightly different values. The detection
  algorithm is the same since 1997, so training and live inputs are consistent. NASA DONKI
  was unavailable during development and only starts in 2010; it is on the roadmap.
- **NOAA RTSW** (`/json/rtsw/rtsw_{wind,mag}_1m.json`): includes several spacecraft at once;
  `source` identifies the spacecraft and `active` marks NOAA's operational one, which is the
  one used for forecasts. The source keeps only ~24 h, so collection must run every hour.
- **NOAA Kp forecast** (`noaa-planetary-k-index-forecast.json`): the source only shows the
  current forecast. We archive **every issued version** (`time` = collection time,
  `target_time` = forecast block), including the `estimated` future blocks.
- **NOAA 3-Day Forecast archive** (NCEI, `…/swpc-products/daily_reports/3day_forecast/`): all
  editions (00:30 and 12:30 UTC) since March 2022. Only blocks that had not started at issue
  time are kept. September–October 2022 are missing from the archive.
- **Kyoto Dst** (`kyoto-dst.json`): real-time, provisional values.
- **GOES** X-ray (two bands), integral protons (≥1 to ≥500 MeV), flare list; **solar regions**
  with McIntosh and Mount Wilson classes and NOAA's C/M/X flare probabilities.

</details>

## 3. Data pipeline

There is no always-on server or database. Everything is precomputed into files, which keeps
the project at zero cost.

```
 GitHub Actions (hourly; triggered by cron-job.org, see §10)
   ├─ futurespace update     connectors download and standardize each source;
   │                         the writer merges, deduplicates and writes Parquet + JSON
   ├─ futurespace predict    applies the published models to the latest data
   └─ futurespace verify     (daily) joins past predictions with observations
                    │
                    ▼
 Hugging Face dataset (data + models, versioned by git commits)
                    │
                    ▼
 FastAPI on Vercel (reads JSON only, cached)  ──▶  API users / website

 GitHub Actions (monthly): futurespace train --publish-if-better
```

- **Idempotent:** running a collection twice duplicates nothing. Values revised by the source
  (e.g. preliminary → definitive Kp) replace the old ones.
- **Fault tolerant:** if one source is down, the others proceed and the error is logged.
- **Partitioning:** by day (1–5 min data), month (OMNI, flares, regions, predictions) or year
  (Kp, Dst, CMEs). Each dataset has a `manifest.json` with its partitions and coverage.

## 4. Methods

### 4.1 Targets

| Question | Index | Horizons *h* |
|---|---|---|
| Will there be a geomagnetic storm, and how strong (NOAA G scale)? | Kp | 3, 6, 12, 24 h |
| How intense will the effect be at low latitudes? | Dst | 1, 3, 6, 12, 24 h |

For each index and horizon there are two models:

- **Regressor:** the value at *t + h*.
- **Classifier:** P(storm at *t + h*), where a storm means **Kp ≥ 5−** (4.67, NOAA G1) or
  **Dst ≤ −50 nT**. An **alert** is raised when the probability exceeds a threshold chosen on
  the validation set (§4.4).

Kp targets refer to the 3-hour block containing *t + h*.

### 4.2 Features

Rule: **only variables that also exist in real time.** The same model trained on OMNI can
therefore run live on the hourly collected data. Version 3 uses 82 features in five groups:

| Group | Features (n) | Rationale |
|---|---|---|
| **Solar wind at L1** (55) | Bz, By, \|B\|, speed, density, temperature, dynamic pressure; Newell coupling `v^(4/3) B_T^(2/3) sin^(8/3)(θ/2)`; VBs = v·max(−Bz, 0); clock-angle sin/cos; lags of 1–6 h, means over 3–24 h and minimum Bz over 3–12 h for Bz, v, n, Newell and VBs | energy entering the magnetosphere now and its trend |
| **Earth's current state** (11) | Dst and its lags, 1 h and 3 h differences, 24 h minimum; last completed Kp block, the previous block, 24 h maximum | whether a storm is already under way |
| **27-day recurrence** (4) | speed one solar rotation ago, and speed, Kp and Dst over the equivalent upcoming 24 h, one rotation back | high-speed streams from coronal holes recur |
| **CMEs** (8) | fastest wide CME (width ≥ 90°) in the last 24/48/72/96 h, number of wide CMEs in 72 h, ballistic arrival estimate (hours to arrival, speed and width of the nearest one) | lead time of hours to days |
| **Calendar** (4) | day of year and UT hour, as sin/cos | Russell–McPherron seasonal effect and diurnal effect |

**Leakage controls** (enforced by automated tests):

- The row for hour *t* only uses data up to the end of hour *t*.
- A Kp value enters only after its 3-hour block has ended.
- A CME exists for the model only **6 h after its onset**, the typical latency of the
  real-time catalog.
- Before the start of the CME catalog, CME features are missing (unknown), not zero.

### 4.3 Model

Gradient-boosted decision trees: scikit-learn `HistGradientBoostingRegressor` and
`HistGradientBoostingClassifier` (version pinned at 1.9.1). This is classical supervised
learning trained from scratch on this dataset; there is no language model involved. Gradient
boosting was chosen because it is strong on tabular data, trains in minutes on a CPU,
handles missing values natively and is inspectable.

| Hyperparameter | Value |
|---|---|
| learning rate | 0.05 |
| max leaf nodes | 31 |
| min samples per leaf | 40 |
| L2 regularization | 1.0 |
| early stopping | off |
| random state | 0 (deterministic) |
| regressor iterations | grid {200, 500, 1000}, chosen by validation RMSE |
| classifier iterations | 400 |
| alert threshold | grid 0.02–0.90 (step 0.01), chosen by validation HSS |

The published v3 bundle contains 9 regressor/classifier pairs: 8,400 trees and 512,400 nodes.

### 4.4 Training procedure

1. **Temporal split, never random:** train 1995-01-01 – 2014-12-31, validation 2015-01-01 –
   2018-12-31, test 2019-01-01 onward.
2. Fit regressors for each `max_iter` in the grid on train and keep the lowest validation RMSE.
3. Fit the classifier on train. On validation, choose the alert threshold that **maximizes HSS**.
   Maximizing TSS gave higher TSS (up to 0.98) but 60–95% false alarms, because TSS barely
   penalizes false alarms for rare events.
4. Refit on train + validation and **evaluate once on test**. These are the reported metrics.
5. **Refit on all data, including the test period** (the current solar cycle), and publish
   that model. The reported metrics remain those of step 4.

### 4.5 Live operation

Every hour, `futurespace predict` builds the same hourly table from real-time data. It uses
hourly means of the operational L1 spacecraft (at least 30 minutes of data required), Kyoto
Dst, NOAA Kp and CACTus real-time CMEs. OMNI and GFZ Kp fill the older hours needed for
recurrence. The forecast is issued for the last complete hour. No forecast is issued without
current solar wind.

### 4.6 Publication gate and retraining

Retraining runs monthly in GitHub Actions. A new model is published only if, on the same test
split, its mean RMSE ratio to the current model is ≤ 1.02 **and** its mean BSS does not
drop by more than 0.02. The model does **not** learn online from its errors. This is
deliberate: a model that changes after every error can degrade without anyone noticing.

## 5. Evaluation protocol

- **Baseline:** persistence ("it will stay the same"): the current Dst, or the last completed
  Kp block. Models are evaluated only where the baseline exists.
- **Climatology:** the storm frequency in the training data, used as the reference for BSS.
- **Metrics** (contingency table: hits *a*, false alarms *b*, misses *c*, correct negatives *d*):

| Metric | Formula | Meaning |
|---|---|---|
| RMSE, MAE | √mean(ŷ−y)², mean\|ŷ−y\| | error of the value (nT or Kp units) |
| POD | a / (a + c) | fraction of storms detected |
| FAR | b / (a + b) | fraction of alerts that were false |
| TSS | POD − b / (b + d) | 1 perfect, 0 no skill |
| HSS | 2(ad − bc) / [(a+c)(c+d) + (a+b)(b+d)] | skill relative to random chance |
| Brier | mean(p − o)² | quality of the probability |
| BSS | 1 − Brier / Brier_climatology | > 0 beats climatology |

## 6. Results

### 6.1 Test period (2019-01-01 – 2026-10-30)

Model `v3-20261106`, 68,600 test hours, with 2,174 storm hours by Dst and 1,584 by Kp:

| Forecast | RMSE model | RMSE persistence | Gain | POD | FAR | TSS | HSS | HSS persistence | BSS | Threshold |
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

RMSE is in nT for Dst and in Kp units for Kp. Gain is the RMSE reduction relative to
persistence. POD, FAR, TSS and HSS are for the classifier alert.

**Findings**

- The forecast value beats persistence at every horizon, with **13–25% lower RMSE**.
- The storm alert is useful up to about **6 h**. At 12–24 h, most alerts are false, and the BSS
  approaches zero for Kp.
- **CME features barely change skill.** The previous feature set without CMEs (v2, 70 features)
  gives BSS within ±0.02 at every horizon. The CACTus catalog does not say whether a CME is
  Earth-directed, and many storms come from high-speed streams instead of CMEs.

### 6.2 What the models rely on (feature-group importance)

For each group of features, all columns in the group are permuted **jointly** on the test
period (20,000 sampled hours, 3 repeats). We measure how much the error increases: RMSE for
the value, Brier score for the storm probability. The share is each group's fraction of the
total increase.

| Model | Solar wind | Earth's state | Recurrence | CMEs | Calendar | RMSE if solar wind permuted |
|---|---|---|---|---|---|---|
| Kp +3 h (value) | **78%** | 17% | 0% | 0% | 5% | 0.76 → 1.34 |
| Kp +3 h (storm probability) | **77%** | 16% | 0% | 1% | 6% | |
| Kp +24 h (value) | **54%** | 12% | 19% | 8% | 8% | 1.15 → 1.26 |
| Kp +24 h (storm probability) | **40%** | 10% | 18% | 15% | 17% | |
| Dst +6 h (value) | 23% | **75%** | 0% | 0% | 1% | 10.3 → 13.4 |
| Dst +6 h (storm probability) | 22% | **76%** | 0% | 0% | 2% | |

Measured with the v3 code on data through 2026-10-30. From the next training run onward, every
run stores this breakdown for all nine models in `/v1/models`.

Interpretation:

- **Kp at short range is driven by the solar wind arriving now.** Bz, speed and coupling
  functions dominate.
- **Dst behaves like a slowly decaying reservoir.** Its current value and trend explain most
  of the next hours, consistent with Burton-type ring-current models.
- **At 24 h, solar memory starts to matter.** Recurrence and CMEs together account for about
  27% of the value model, because the L1 solar wind is less informative about the next day.

### 6.3 Comparison with NOAA (2022–2026)

Using the NCEI archive of NOAA's 3-Day Forecast, both forecasts were compared with observed
Kp (GFZ) on the **same blocks and lead times**. For each NOAA edition, the model uses only
data up to the previous full hour, and it uses the model trained up to 2018, which never saw
this period.

| Lead time | Blocks | MAE model | MAE NOAA | Alert HSS model | Alert HSS NOAA |
|---|---|---|---|---|---|
| ~6 h | 2,534 | **0.72** | 1.16 | **0.40** | 0.24 |
| ~12 h | 2,534 | **0.80** | 1.18 | **0.21** | 0.16 |
| ~24 h | 2,534 | **0.90** | 1.16 | 0.08 | **0.16** |

- The model has lower error in the Kp value at all three lead times.
- For storm alerts, the model is better at 6 and 12 h and NOAA is better at 24 h. NOAA
  forecasters analyze solar imagery and CME direction, which matters most at longer lead times.
- NOAA's error is about 1.15 Kp units regardless of lead time. About 82% of NOAA's G1 alerts in
  this period were not confirmed.
- **Caveat:** in this hindcast the model uses OMNI and definitive Kp, which are slightly cleaner
  than real-time data. Live verification (§6.4) tests whether the advantage holds.

### 6.4 Live verification

Each day, `futurespace verify` joins every issued forecast with the value observed later. It
stores the error, the persistence error at issue time and, for Kp, the error of the NOAA
forecast issued up to the same moment. Rolling 7- and 30-day summaries are at
`GET /v1/verification/summary`. Live forecasts started in November 2026, so the sample is
still small. In the first 10 verified Kp +3 h forecasts, all without storms, the MAE was 0.49
for the model, 0.50 for persistence and 1.45 for NOAA.

### 6.5 Worked example: forecast issued 2026-11-07 19:00 UTC

Inputs at 18:00 UTC: solar wind 653 km/s (rising from 522 km/s five hours earlier), Bz
turning south to −3.6 nT, Newell coupling 15,490 (3-hour mean 7,330), last
completed Kp block 3.0, Dst −5 nT.

The Kp +3 h regressor starts from the training mean, **1.83**, and adds the outputs of 1,000
trees. Tree 1 follows this path: `kp_known > 2.17` → `kp_known ≤ 3.83` → `newell > 6,989` →
`newell > 9,673`, which adds **+0.094**. The first 100 trees add +2.51 in total, and all
1,000 trees add +2.47. The forecast is therefore **Kp 4.30** for the 21–24 UTC block.

The classifier gave a storm probability of **27.6%**, above the 0.21 threshold chosen on
validation, so a G1 alert was raised. Persistence would have said Kp 3.0. This forecast is
recorded in the `predictions` dataset and will be scored in `verification`.

## 7. Limitations

- **Storm onset is hard to predict.** The model mainly sees the solar wind at L1, about 1 h
  before it reaches Earth, and CME features carry little information without Earth-directedness.
- **Extreme storms are underestimated.** Tree ensembles cannot extrapolate beyond the training
  range, and events like May 2024 (Dst −406 nT) are very rare in the record.
- **Training/operation mismatch.** OMNI is propagated to the bow shock, while live data is the
  raw L1 measurement, roughly 45 minutes earlier. This makes live forecasts slightly conservative.
- **Real-time data quality.** Provisional Dst and spacecraft changes at L1 (SOLAR-1, IMAP,
  ACE) can introduce jumps that are absent in OMNI.
- **Feature importance is a model property, not causation.** Permutation importance shows what
  the model uses, and correlated groups share credit.

## 8. Reproducibility

Every training run stores a full record (`report.json`) next to its models in the Hugging
Face dataset, and an index of all runs (`models/index.json`). This includes runs rejected
by the publication gate, with the reason.

| Field | Content |
|---|---|
| `reproducibility.code` | GitHub commit used for training, and whether there were uncommitted changes |
| `reproducibility.data` | Hugging Face dataset commit that was read, plus rows and coverage per source |
| `reproducibility.hyperparameters` | tree parameters, search grids, horizons, storm definition, importance settings |
| `reproducibility.environment` | Python, scikit-learn, numpy and pandas versions |
| `splits`, `features`, `feature_groups` | temporal split, ordered feature list, group membership |
| `models[]` | per model: chosen iterations, threshold, rows, test metrics, feature-group importance |

```bash
curl https://future-space-six.vercel.app/v1/models                # all runs, metrics, importance
curl https://future-space-six.vercel.app/v1/models/v3-20261106    # full record of one run
```

**Reproducing a model.** Training is deterministic (`random_state=0`, no early stopping).
Check out the recorded commit, install the pinned versions from `requirements-pipeline.txt`
and read the data at the recorded revision (`HuggingFaceStorage(repo, revision=<sha>)`). With
the same library versions, this produces the same model. Model `v3-20261106` predates these
records, so its code commit and data revision are empty. Its data ends at 2026-10-30 14:00 UTC
and it was published in dataset commit `20c2da27`.

Files in the dataset:

```
parquet/<dataset>/<partition>.parquet         training and analysis
json/<dataset>/<partition>.json               read by the API
json/<dataset>/manifest.json                  partitions and coverage
json/catalog.json                             full catalog
models/current.json                           model version in use
models/index.json                             history of all training runs
models/<version>/bundle.joblib                trained models
models/<version>/report.json                  full record of that version
models/rejected/<version>-<time>/report.json  runs rejected by the gate (no models)
```

## 9. Using the data and the API

**Parquet files** can be read directly, in Python, Kaggle, Colab or DuckDB:

```python
import pandas as pd  # pip install pandas pyarrow huggingface_hub

omni = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/omni-hourly/")
kp = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/kp-gfz/")
cmes = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/cme-cactus/")
```

| Size (measured) | Parquet | CSV | JSON |
|---|---|---|---|
| OMNI 1963–present (560k rows) | 25 MB | 60 MB | 148 MB |
| GFZ Kp 1932–present (277k rows) | 2.6 MB | — | 15 MB |
| L1 magnetic field, 1 day, 3 spacecraft | ~50 KB | — | ~340 KB |

**API** (base `https://future-space-six.vercel.app`, interactive docs at `/docs`):

| Endpoint | Returns |
|---|---|
| `GET /v1/datasets` | datasets and their coverage |
| `GET /v1/datasets/{id}` | metadata, fields, units, license, coverage |
| `GET /v1/datasets/{id}/latest` | latest values |
| `GET /v1/datasets/{id}/data?start=&end=&fields=&format=` | time series in a range (JSON or CSV) |
| `GET /v1/datasets/predictions/latest` | latest model forecasts |
| `GET /v1/compare/kp?date=YYYY-MM-DD` | model × NOAA × observed Kp, block by block |
| `GET /v1/compare/dst?date=YYYY-MM-DD` | model forecasts × observed Dst, hourly |
| `GET /v1/verification/summary` | forecast errors over the last 7 and 30 days |
| `GET /v1/models` | model history: versions, metrics, feature-group importance |
| `GET /v1/models/{id}` | full record of a training run |

```bash
# May 2024 superstorm in OMNI
curl "https://future-space-six.vercel.app/v1/datasets/omni-hourly/data?start=2024-05-10&end=2024-05-12&fields=bz_gsm,speed,kp,dst"
```

Each dataset has a maximum window per request (`max_query_days`, e.g. 3 days for 1-minute
data). For large volumes, use the Parquet files. Rate limit: 60 requests per minute per IP
(`429` with `Retry-After`). Unknown query parameters are rejected with `400`. See
[SECURITY.md](SECURITY.md).

## 10. Operations

**Run locally** (Python 3.11+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,hf]"

futurespace update                                   # collect real-time data → ./data
futurespace backfill omni-hourly --start 1963-01-01  # full OMNI history (~4 min)
futurespace backfill kp-gfz --start 1932-01-01       # full Kp history (~30 s)
futurespace backfill cme-cactus --start 1997-01-01   # CMEs since 1997 (~40 s)
futurespace train                                    # train and evaluate (~10 min) → ./models
futurespace train --no-cme --out models-no-cme       # ablation without CME features
futurespace predict                                  # forecast from current data
futurespace serve --reload                           # API at http://localhost:8000/docs
pytest -q
```

| Variable | Use | Default |
|---|---|---|
| `FS_STORAGE` | where the pipeline writes: `local`, `hf` or `s3` | `local` |
| `FS_DATA_DIR` | folder for `local` | `data` |
| `FS_HF_REPO` | Hugging Face dataset, e.g. `Strutf/future-space` | — |
| `HF_TOKEN` | write token (pipeline only, kept in GitHub secrets) | — |
| `FS_DATA_URL` | alternative base URL for the API to read JSON | — |
| `FS_RATE_LIMIT_PER_MINUTE` | API requests per minute per IP (`0` disables) | `60` |

**Zero-cost deployment.** The pipeline runs on GitHub Actions, which is free for public
repositories. Data and models are stored in a public Hugging Face dataset, written through a
fine-grained token that exists only in GitHub secrets. The API and website run on Vercel
Hobby and read only public data, so they need no token.

**Scheduling.** GitHub's native `on.schedule` never fired in this repository, even at 5-minute
intervals. Runs are therefore triggered externally by [cron-job.org](https://cron-job.org),
which calls GitHub's `workflow_dispatch` API:

| Job | Schedule (UTC−3) | Workflow | Body |
|---|---|---|---|
| ingest realtime | `23 * * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"realtime"}}` |
| ingest daily | `47 0 * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"daily"}}` |
| monthly training | `41 1 1 * *` | `train.yml` | `{"ref":"main"}` |

Each job sends a `POST` with `Authorization: Bearer <fine-grained token with Actions: read and
write>`; the expected response is `204`. The native schedule stays in the workflows as a
fallback. All writing jobs share a concurrency group, so they never write at the same time.

## 11. Website

https://future-space-six.vercel.app is served by the API itself, without a build step, in
Portuguese, English, Spanish and French. It has a hand-drawn chalkboard look aimed at a
general audience.

- **Forecasts:** model × NOAA × observed Kp (default tab), model storm probability, and hourly
  Dst, for yesterday, today and tomorrow.
- **About:** how a solar storm happens, what Kp, Dst and CMEs are, and every formula used.
- **Interactive** (runs in the browser, no login):
  - a CME simulator using a reduced monotonic model (speed, density, VBs) exported from
    scikit-learn to JSON, with a test guaranteeing identical outputs;
  - historical storm replays (1989, 2003, 2015, 2024) with what the model trained up to 2018
    would have forecast;
  - an aurora map based on a geomagnetic dipole and a visibility line calibrated on NOAA's
    aurora maps (≈ 66° − 1.5° × Kp magnetic latitude).

Texts live in `src/futurespace/api/web/locales/{pt,en,es,fr}.json`. Tests ensure all
languages have the same keys and placeholders.

## 12. Code structure

```
src/futurespace/
├── catalog.py          dataset metadata (fields, units, licenses, schedules)
├── partitions.py       file layout and time utilities
├── storage.py          backends: local, Hugging Face, S3, HTTP
├── model_index.py      model history (models/index.json), readable without ML libraries
├── cli.py              `futurespace` command
├── pipeline/           HTTP client, writer (merge/deduplicate), one connector per source
├── ml/
│   ├── features.py     hourly table, features, targets, feature groups (leakage-free)
│   ├── train.py        training, temporal validation, evaluation, importance, records
│   ├── metrics.py      RMSE, MAE, POD, FAR, TSS, HSS, Brier
│   ├── serve.py        live predictions and model publication
│   ├── verify.py       forecast vs. observed
│   └── export_web.py   browser models and storm replays for the website
└── api/
    ├── app.py          FastAPI (no pandas, fits serverless)
    ├── compare.py      model × NOAA comparison for the dashboard
    └── web/            pages, static files and translations
api/index.py, vercel.json    Vercel entry point
.github/workflows/           ingest, train, tests
requirements-pipeline.txt    pinned versions used by the workflows that hold the token
```

**Adding a source:** describe the dataset in `catalog.py`, write a function
`(client, start, end) -> DataFrame` in `pipeline/connectors/` and register it in
`connectors/__init__.py`. The writer and the API then serve it automatically.

## 13. Roadmap

- [x] Collection of 13 datasets, public API, open Parquet/JSON dataset
- [x] Kp and Dst forecasts 1–24 h ahead (value, probability, alert), against persistence
- [x] CME features from CACTus
- [x] Hourly live forecasts, daily verification, comparison with NOAA
- [x] Monthly retraining with a publication gate
- [x] Reproducible model records and feature-group importance
- [x] Website in four languages with interactive explainers
- [ ] CME direction and arrival (NASA DONKI / WSA–Enlil) for longer lead times
- [ ] New sources: INTERMAGNET magnetometers (Vassouras and Tatuoca, Brazil), INPE/EMBRACE,
      ionospheric TEC
- [ ] Flare forecasting (SDO/HMI SHARP parameters), event linking across sources (which CME
      caused which storm), anomaly detection
- [ ] Probabilistic calibration analysis (reliability diagrams) and storm-onset-specific metrics

## 14. How to cite

If you use the data, models or results, please cite this repository (GitHub's
*Cite this repository* button reads [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{florido_future_space_2026,
  author  = {Flórido, Victor},
  title   = {Future Space: open multi-source space-weather data and machine-learning
             forecasts of geomagnetic storms},
  year    = {2026},
  url     = {https://github.com/Strutf/future-space},
  version = {0.1.0}
}
```

To cite a specific model, include its version and record, e.g. `v3-20261106` and
`https://future-space-six.vercel.app/v1/models/v3-20261106`.

Please also cite the original data sources (§15).

## 15. License and data credits

Code: MIT. Each dataset keeps the license of its original source, given in the catalog
(`/v1/datasets/{id}`).

- **OMNI:** King, J. H., & Papitashvili, N. E. (2005). Solar wind spatial scales in and
  comparisons of hourly Wind and ACE plasma and magnetic field data. *JGR*, 110, A02104.
  doi:10.1029/2004JA010649. NASA GSFC SPDF / OMNIWeb.
- **Kp:** Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O., & Morschhauser, A. (2021).
  The geomagnetic Kp index and derived indices of geomagnetic activity. *Space Weather*, 19,
  e2020SW002641. doi:10.1029/2020SW002641. Data: doi:10.5880/Kp.0001 (CC BY 4.0).
- **Dst:** World Data Center for Geomagnetism, Kyoto, Nose, M., Iyemori, T., Sugiura, M., &
  Kamei, T. (2015). Geomagnetic Dst index. doi:10.17593/14515-74000.
- **CACTus:** Robbrecht, E., & Berghmans, D. (2004). Automated recognition of coronal mass
  ejections (CMEs) in near-real-time data. *A&A*, 425, 1097–1106.
  doi:10.1051/0004-6361:20041302. SIDC, Royal Observatory of Belgium.
- **NOAA SWPC** real-time products and the NCEI forecast archive are public domain (U.S.
  Government).
- **Newell coupling:** Newell, P. T., et al. (2007). A nearly universal solar wind–magnetosphere
  coupling function inferred from 10 magnetospheric state variables. *JGR*, 112, A01206.
  doi:10.1029/2006JA012015.
- **scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  *JMLR*, 12, 2825–2830.
