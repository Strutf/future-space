# Aurora probability research

Goal: replace the uncalibrated aurora-visibility score (v0) with a calibrated probability.

## Pipeline

The research scripts run on demand, locally or on any machine; they are not part of the
hourly GitHub Actions pipeline. Only live data collection runs there. Outputs are stored on
Hugging Face under `research/aurora/`.

| Step | Script | Output |
|---|---|---|
| 1. Ground truth | `sample_themis.py <site> <start> <end> <n> <out>`, one task of `themis_plan.json` | `themis/<site>_<start>_<end>/` |
| 2. Labels and context | `build_dataset.py <hf_repo> <work_dir> <out_dir>` | `aurora_samples.parquet`, `cloud_samples.parquet` |
| 3. Probability fit and report | `fit_probability.py <out_dir>` | `report.json` |

Collected so far: 30 of the 108 tasks in `themis_plan.json`. These cover 6 cameras and the
winters 2007–08, 2008–09, 2011–12, 2014–15, 2016–17 and 2021–22 (not every camera in every
winter), with up to 200 frames each.

**Step 1, ground truth.** The script draws random dark, clear-sky moments. Clear sky comes
from the University of Calgary cloud classification (AuroraX, every 10 minutes). For each
moment it downloads one raw frame.

**Step 2, labels and context.**

- The image classifier (`classifier.py`) labels each frame as aurora, clear or other. It
  uses pretrained ResNet-18 features and logistic regression, trained on OATH (Clausen &
  Nickisch, 2018, CC BY 4.0).
- In validation grouped by night (5 folds, 81 nights), accuracy is 89.6% overall and 91.2%
  for aurora vs. clear. Fine-tuning the last ResNet block gave the same result (89.7% overall,
  90.1% aurora vs. clear).
- Each frame is joined with Kp (GFZ), solar wind (OMNI), dipole magnetic latitude and
  magnetic local time, and Sun and Moon positions.
- Calgary's cloud labels are joined with ERA5 cloud cover from Open-Meteo, for the cloud
  calibration.

**Step 3, probability fit and report.** The fit covers:

- P(aurora | Kp, magnetic latitude, magnetic local time) for dark, clear-sky moments;
- P(clear sky | ERA5 cloud cover).

Both are validated by night with the Brier skill score and reliability diagrams, and compared
with the backend's v0 score. The results are written to `results/report.json`.

## Results (v1, 30 tasks)

The labelled samples and the full report are in `results/`: `aurora_samples.parquet` and
`report.json`. The cloud samples (12 MB) can be rebuilt with `build_dataset.py`.

### Aurora

The data cover 4,896 dark, clear-sky frames from 1,625 nights at 6 cameras, between 61° and
73° magnetic latitude. Aurora appears in 33.8% of the frames. Another 1,089 frames were
excluded because they were classified as "other" (Moon or residual cloud).

Cross-validated by night:

| Model | Brier | BSS vs. climatology |
|---|---|---|
| v0 score (backend) | 0.302 | −0.35 |
| logistic regression | 0.172 | +0.23 |
| gradient boosting (monotonic in Kp) | 0.154 | +0.31 |

Reliability of the gradient-boosting model:

| Forecast | 0.03 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.84 | 0.93 |
|---|---|---|---|---|---|---|---|---|---|---|
| Observed | 0.04 | 0.18 | 0.29 | 0.38 | 0.47 | 0.52 | 0.62 | 0.66 | 0.85 | 0.83 |

The v0 score is overconfident: when it says 0.98, aurora is seen 57% of the time.

Observed aurora frequency by Kp:

| Kp | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| frequency | 13% | 28% | 39% | 49% | 64% | 69% |

### Clouds

The data cover 246,431 dark hours, of which the camera saw a clear sky in 44.1%.

| Model | Brier | BSS vs. climatology |
|---|---|---|
| v0 rule, 1 − cloud cover | 0.265 | −0.08 |
| logistic regression on ERA5 cloud layers | 0.201 | +0.19 |

Observed clear-sky frequency by ERA5 cloud cover:

| ERA5 cloud cover | 0% | 50% | 90% | 100% |
|---|---|---|---|---|
| camera sees a clear sky | 81% | 57% | 36% | 25% |

### Caveats

- **Camera vs. naked eye.** The cameras are more sensitive than the human eye, so the model
  estimates P(aurora detectable by an all-sky camera), which is an upper bound for naked-eye
  visibility.
- **Latitude coverage.** All cameras are between 61° and 73° magnetic latitude. Results at
  lower magnetic latitudes (London, New York, southern Brazil) would be extrapolation.
- **Fort Smith 2021–22.** This winter shows aurora in 84% of frames, against 32–43% in other
  winters with the same imager (themis10). This needs checking for a change in exposure or
  sensor.
- **Reanalysis vs. forecast.** ERA5 is a reanalysis. A forecast archive would give a fairer
  calibration of the live cloud forecast.

## Requirements

The research scripts need `torch`, `torchvision`, `pillow`, `scikit-learn` and `pandas`.
These are not project dependencies.
