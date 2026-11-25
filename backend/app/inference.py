from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "densenet121_063A.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF"}

# Dimension limits for a safe resize to 224x224. Training images (BreakHis) are 700x460,
# i.e. aspect ratio ~1.52 and ~3x downscale. Tunable: widen only after checking predictions.
MIN_SIDE = 224          # smaller -> upscaled, blurry, no real cell detail
MAX_SIDE = 2048         # larger -> >9x downscale, cell detail lost
MAX_ASPECT_RATIO = 2.0  # long side / short side; beyond this the squash distorts morphology

with METADATA_PATH.open("r", encoding="utf-8") as f:
    MODEL_METADATA = json.load(f)

IMG_HEIGHT = int(MODEL_METADATA["input"]["height"])
IMG_WIDTH = int(MODEL_METADATA["input"]["width"])
THRESHOLD = float(MODEL_METADATA["decision_threshold"])
MODEL_ID = f'{MODEL_METADATA["experiment_id"]}-{MODEL_METADATA["backbone"]}-group-balanced'

_model: tf.keras.Model | None = None


class ImageValidationError(ValueError):
    pass


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                "Package the Experiment 06.3A checkpoint first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def check_dimensions(width: int, height: int) -> None:
    """Reject images that cannot be resized to the model input without distortion or detail loss.
    Runs on header size only, before pixels are decoded."""
    size = f"{width}x{height}"
    short, long = min(width, height), max(width, height)
    if short < MIN_SIDE:
        raise ImageValidationError(
            f"Image is too small ({size}). Both sides must be at least {MIN_SIDE} px; "
            "upload the original microscope image, not a thumbnail."
        )
    if long > MAX_SIDE:
        raise ImageValidationError(
            f"Image is too large ({size}). The longest side must be at most {MAX_SIDE} px, "
            "otherwise cell detail is lost when it is reduced to 224x224. "
            "Upload a smaller field of view or downscale it (e.g. 700x460, like the training data)."
        )
    if long / short > MAX_ASPECT_RATIO:
        raise ImageValidationError(
            f"Image shape is too stretched ({size}, ratio {long / short:.2f}:1). "
            f"The ratio must be at most {MAX_ASPECT_RATIO:g}:1, otherwise tissue gets squashed "
            "when resized to 224x224. Crop it closer to square (e.g. 700x460)."
        )


def preprocess_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Image exceeds the 12 MB upload limit.")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError("Only JPEG, PNG, and TIFF images are supported.")
            check_dimensions(*image.size)
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a valid readable image.") from exc

    # Match training exactly: any input size -> TF bilinear resize with antialias to 224x224
    # (values stay in 0..255), then DenseNet preprocess_input. No extra 0..1 rescale.
    tensor = tf.image.resize(array, (IMG_HEIGHT, IMG_WIDTH), method="bilinear", antialias=True)
    tensor = tf.keras.applications.densenet.preprocess_input(tensor)
    return np.expand_dims(tensor.numpy(), axis=0)


def predict_bytes(raw: bytes, filename: str) -> dict[str, Any]:
    model = load_model()
    batch = preprocess_image(raw)
    probability = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
    predicted = "malignant" if probability >= THRESHOLD else "benign"
    return {
        "filename": filename,
        "predicted_class": predicted,
        "malignant_probability": probability,
        "threshold": THRESHOLD,
        "threshold_decision": f"malignant_probability >= {THRESHOLD:.2f}",
        "model_id": MODEL_ID,
    }


def model_status() -> bool:
    return MODEL_PATH.exists()
