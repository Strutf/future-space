# Aurora probability research

Goal: replace the uncalibrated aurora-visibility score (v0) with a calibrated probability.

## Pipeline

Everything runs on GitHub Actions, and all outputs are stored on Hugging Face under
`research/aurora/`.

| Step | Workflow / script | Trigger |
|---|---|---|
| 1. Ground truth | `aurora-sample` → `sample_themis.py` | after every `ingest` run, processing the next 5 pending tasks of `themis_plan.json` (108 tasks: 6 THEMIS cameras × winters 2007–2024, 200 frames each) |
| 2. Labels and context | `aurora-label` → `build_dataset.py` | after every `aurora-sample` run that added data |
| 3. Probability fit and report | `aurora-label` → `fit_probability.py` | same as step 2 |

**Step 1, ground truth.** The script draws random dark, clear-sky moments. Clear sky comes
from the University of Calgary cloud classification (AuroraX, every 10 minutes). For each
moment it downloads one raw frame. A winter without data is recorded with an `EMPTY` marker,
so its task does not return to the queue.

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

## Pilot result (Fort Smith, winter 2015–16, 229 frames, 32 nights)

| Model | Brier | BSS vs. climatology |
|---|---|---|
| v0 score (backend) | 0.378 | −0.52 |
| logistic, cross-validated by night | 0.155 | +0.38 |
| gradient boosting, cross-validated by night | 0.137 | +0.45 |

Observed aurora frequency by Kp:

| Kp | 0 | 1 | 2 | 5 |
|---|---|---|---|---|
| frequency | 3% | 17% | 72% | 91% |

The v0 score is overconfident at this magnetic latitude (66.5°).

## Requirements

The research scripts need `torch`, `torchvision`, `pillow`, `scikit-learn` and `pandas`.
These are not project dependencies.
