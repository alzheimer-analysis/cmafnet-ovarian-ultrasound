# CMAFNet: Multimodal Ovarian Tumor Classification

This repository contains the code implementation for CMAFNet, a multimodal diagnostic model for preoperative three-class ovarian tumor classification using ultrasound images and routinely available clinical variables.

## Repository Scope

This public repository is intended to contain code only.

It does not include patient-level ultrasound images, clinical records, pathology labels, ROI masks, trained checkpoints, prediction outputs, or manuscript submission files. These materials may contain sensitive health information or unpublished research material and should not be uploaded to a public GitHub repository.

Recommended public files:

- `cmafnet/`
- `experiments/`
- `requirements.txt`
- `README.md`
- `.gitignore`

Excluded private materials:

- `data/`
- `artifacts/splits/`
- `artifacts/checkpoints/`
- `artifacts/results/`
- `BSPC_submission/`
- `elsarticle/`
- raw figures, generated manuscript figures, and local LaTeX build files

## Method Overview

CMAFNet combines:

- a hybrid CNN-Transformer ultrasound image encoder
- a clinical MLP encoder for structured biomarker and demographic variables
- bidirectional cross-modal attention
- gated multimodal fusion
- O-RADS-guided auxiliary supervision during training
- class-weighted focal loss for imbalanced three-class pathology prediction

The three prediction classes are:

- benign ovarian tumor
- malignant ovarian tumor
- borderline ovarian tumor

## Input Variables

The clinical branch expects the following variables:

- age
- menopausal status
- vaginal bleeding
- AFP
- CA125
- CA199
- CEA
- HE4
- maximum tumor diameter

O-RADS is not used as a primary input feature in the leakage-controlled model. It is used only as an auxiliary training target when available.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The code uses PyTorch, torchvision, timm, scikit-learn, scipy, xgboost, Pillow, matplotlib, and PyYAML.

## Project Layout

```text
cmafnet/
  baselines/          comparator models
  clinical_scores/    O-RADS and ROMA utilities
  encoders/           image and clinical encoders
  fusion/             cross-modal attention and gated fusion
  losses/             focal and multitask losses
  metrics/            classification, calibration, DCA, bootstrap, tests
  models/             CMAFNet architecture
  train/              training and checkpoint utilities
  viz/                Grad-CAM, ROC, confusion matrix, t-SNE utilities

experiments/
  prepare_cohort.py
  phase_a_fit.py
  phase_b_validate.py
  phase_c_external.py
  phase_d_baselines.py
  phase_e_ablation.py
  phase_f_interpret.py
  phase_g_stats.py
```

## Running Experiments

Set the repository root as `PYTHONPATH` before running experiment stages:

```bash
export PYTHONPATH=/path/to/repository
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="E:\part_time_paper\061607_Ultrasound direction_zone 2"
```

Run stages separately:

```bash
python experiments/prepare_cohort.py
python experiments/phase_a_fit.py
python experiments/phase_b_validate.py
python experiments/phase_c_external.py
python experiments/phase_d_baselines.py
python experiments/phase_e_ablation.py
python experiments/phase_f_interpret.py
```

This repository intentionally does not provide a one-command reproduction script. Each stage should be reviewed and executed separately.

## Data Availability

The clinical and ultrasound data are not included in this public code repository. The original study data contain sensitive medical information and are available only according to the manuscript data availability statement and institutional requirements.

Users who want to reproduce the full study should prepare their own de-identified dataset with the same schema and ensure that all use complies with local ethics approval, data protection rules, and institutional policies.

## Privacy Notice

Do not commit:

- patient images
- clinical CSV files
- pathology labels linked to patient IDs
- ROI masks
- internal patient or hospital identifiers
- trained weights derived from non-public medical data
- prediction files containing case-level outputs

Before making the repository public, run a final check to confirm that only code and non-sensitive configuration files are tracked.

## Intended Use

This code is provided for academic research and method reproduction only. It is not a medical device and must not be used as a standalone clinical diagnostic system.

