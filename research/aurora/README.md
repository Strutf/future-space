# Aurora probability research

Goal: replace the uncalibrated aurora-visibility score (v0) with a calibrated probability.

1. **Ground truth.** `sample_themis.py` draws random dark, clear-sky moments from THEMIS
   all-sky cameras. Clear sky comes from the University of Calgary cloud classification
   (AuroraX). The script downloads one raw frame for each moment. The `aurora-sample`
   workflow runs the plan in `themis_plan.json` on GitHub Actions and stores the frames on
   Hugging Face under `research/aurora/themis/`.
2. **Image classifier.** `classifier.py` labels each frame as aurora, clear or other. It uses
   pretrained ResNet-18 features and logistic regression, trained on the OATH dataset
   (Clausen & Nickisch, 2018, CC BY 4.0).

   Validation grouped by night (5 folds), 81 nights:

   | Task | Accuracy |
   |---|---|
   | overall | 89.6% |
   | aurora vs. clear | 91.2% |

   Fine-tuning the last ResNet block gave the same result (89.7% overall, 90.1% aurora vs.
   clear).
3. **Probability model** (next step): fit P(aurora | Kp, magnetic latitude, magnetic local
   time, …) on the labelled random samples, then check it with a reliability diagram and the
   Brier score.

The classifier needs `torch`, `torchvision`, `pillow`, `scikit-learn` and `pandas`. These
are not project dependencies.
