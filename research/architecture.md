# System architecture

## Current target

Hybrid UI: dark navigation/sidebar with a clean, light workspace.

```text
React + Vite frontend
        |
        | REST / JSON + multipart image uploads
        v
FastAPI backend
  - upload validation
  - training-matched preprocessing
  - inference
  - Grad-CAM
  - model metadata
        |
        v
TensorFlow / Keras DenseNet121 checkpoint
```

## Backend responsibilities

- Load the Keras checkpoint once at process startup.
- Validate image format, byte size, and pixel count.
- Convert to RGB and resize to 224×224.
- Apply DenseNet121 `preprocess_input`; do not add an extra 0–1 rescaling.
- Return malignant probability, thresholded class, threshold, and model ID.
- Generate Grad-CAM overlays from the final spatial DenseNet feature map.

## Frontend responsibilities (next phase)

- Overview and research metrics.
- Single and batch image upload.
- Per-image prediction results.
- Grad-CAM visualization with explicit caveat.
- Model evaluation and methodology pages.
- Research notes and limitations.

## Deployment

- Local: Vite dev server + Uvicorn.
- Cloud: containerized backend and static frontend hosting; model artifact supplied securely through release storage or Git LFS where appropriate.
