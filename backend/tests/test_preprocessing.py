import io
import numpy as np
import pytest
from PIL import Image

from app.inference import preprocess_image, ImageValidationError


def png_bytes():
    image = Image.new("RGB", (700, 460), (120, 80, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_shape_and_dtype():
    batch = preprocess_image(png_bytes())
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_empty_upload_rejected():
    with pytest.raises(ImageValidationError):
        preprocess_image(b"")


def test_non_image_rejected():
    with pytest.raises(ImageValidationError):
        preprocess_image(b"not an image")


def test_large_image_resized_to_model_input():
    buffer = io.BytesIO()
    Image.new("RGB", (1080, 1080), (200, 100, 150)).save(buffer, format="JPEG")
    batch = preprocess_image(buffer.getvalue())
    assert batch.shape == (1, 224, 224, 3)


@pytest.mark.parametrize("size, reason", [
    ((100, 100), "too small"),
    ((4000, 3000), "too large"),
    ((2000, 500), "too stretched"),
])
def test_unsuitable_dimensions_rejected(size, reason):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    with pytest.raises(ImageValidationError, match=reason):
        preprocess_image(buffer.getvalue())
