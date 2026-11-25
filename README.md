# Breast Cancer Histopathology AI

An experimental research application for binary classification of breast histopathology images (benign vs malignant) using a fine-tuned DenseNet121 model trained on the BreakHis-derived dataset split used in Experiment 06.3A.

> **Research use only. Not a medical device.** This application must not be used to diagnose, rule out, screen for, or make treatment decisions about breast cancer. Outputs require interpretation by qualified professionals.

## Current baseline

| Item | Value |
|---|---|
| Experiment | 06.3A — Candidate-group-balanced DenseNet121 |
| Backbone | DenseNet121, ImageNet initialization |
| Input | RGB, resized to 224 × 224 |
| Preprocessing | `tf.keras.applications.densenet.preprocess_input` after resize |
| Precision | FP32 |
| Batch size during training | 8 |
| Training sampler | Class probability proportional to training image counts; uniform candidate group within class; uniform image within group; replacement |
| Fine-tuning | Frozen backbone head training, then final DenseNet block fine-tuning; BatchNorm layers frozen |
| Validation-selected threshold | 0.88 (latest rerun; stored in `backend/app/model/model_metadata.json`) |

Latest reported development-test metrics at threshold 0.88: accuracy 87.91%, ROC-AUC 91.29%, malignant sensitivity 96.00%, benign specificity 71.02%, malignant precision 87.38%, F1 91.48%, balanced accuracy 83.51%.

These are exploratory results from a repeatedly inspected test partition. They are not a fresh independent final evaluation. Candidate-group identifiers parsed from filenames must be verified against official BreakHis patient metadata before describing the split as definitively patient-independent.

## Repository layout

```text
backend/
  app/
    main.py                 FastAPI application
    schemas.py              API response models
    inference.py            Model loading, preprocessing, prediction
    explainability.py       Grad-CAM implementation
    model/
      model_metadata.json   Baseline configuration and metrics
      README.md             Where to put the model weights
  scripts/
    package_model.py        Copy and SHA-256 verify a trained .keras checkpoint
  tests/
    test_preprocessing.py
frontend/                   React application (next implementation phase)
research/
  baseline.md               Research baseline and limitations
  model_card.md             Intended use, metrics, caveats
  architecture.md           System design
sample_images/              Add only legally redistributable examples
```

## 1. Add the trained model checkpoint

The model weights are deliberately **not included** in this starter archive. The `.keras` checkpoint remains on the machine where Experiment 06.3A was trained.

Copy the final checkpoint from the training output directory:

```text
experiment_06_3A_group_balanced_outputs/models/
  Experiment_06_3A_DenseNet121_GroupBalanced_best.keras
```

Place it at:

```text
backend/app/model/densenet121_063A.keras
```

Or use the packaging script:

```bash
python backend/scripts/package_model.py \
  --source "/path/to/Experiment_06_3A_DenseNet121_GroupBalanced_best.keras"
```

The script copies the model and writes a SHA-256 checksum file. It never retrains or modifies the original checkpoint.

## 2. Run the API locally

Python 3.10 or 3.11 is recommended. Create an environment and install dependencies:

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

Health endpoint: http://127.0.0.1:8000/health

The model is loaded once at application startup. If weights are absent, the API starts in an unavailable-model state and returns HTTP 503 for inference routes.

## 3. API endpoints

- `GET /health` — service and model readiness
- `GET /model-info` — baseline model metadata
- `POST /predict` — one image
- `POST /predict/batch` — multiple images (bounded batch)
- `POST /explain` — prediction plus Grad-CAM overlay

Accepted image formats: JPEG, PNG, TIFF. Upload size and pixel count are bounded. The API does not persist uploaded images.

## 4. Reproducibility and scientific boundaries

- Inference preprocessing must match training: decode RGB, resize to 224×224 with antialiasing, then DenseNet preprocessing. Do not rescale to `[0, 1]` separately.
- The validation-selected threshold is 0.88 for the latest rerun. A threshold is an operating point, not a model confidence guarantee.
- An arbitrary user image may differ from BreakHis in staining, magnification, scanner, preparation, and acquisition. The app will display an out-of-distribution warning.
- Grad-CAM visualizes model sensitivity, not a verified biological explanation.
- Do not use the previously inspected test partition to tune model choices and then present it as an untouched final holdout.
- Before publication, verify BreakHis dataset licensing/redistribution terms and candidate-group-to-patient mapping.

## Development status

- [x] Baseline configuration documented from Experiment 06.3A notebook
- [x] Model packaging/checksum workflow
- [x] FastAPI inference skeleton and Grad-CAM endpoint
- [ ] Add trained checkpoint and run inference parity checks
- [ ] React hybrid dashboard and image upload workflow
- [ ] Evaluation visualizations and research pages
- [ ] Docker, CI, security review, deployment
