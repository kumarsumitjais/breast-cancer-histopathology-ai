from __future__ import annotations

import io
import numpy as np
import tensorflow as tf
from PIL import Image

from .inference import (
    IMG_HEIGHT,
    IMG_WIDTH,
    preprocess_image,
    load_model,
    predict_bytes,
)


def _find_densenet_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == "densenet121":
            return layer

    raise RuntimeError("Could not locate the nested DenseNet121 backbone.")


def _build_gradcam_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a connected Grad-CAM graph for the nested DenseNet121 model.

    The backbone and classification-head layers are reused directly.
    No model weights are modified.
    """
    backbone = _find_densenet_backbone(model)

    # Use the final spatial activation of DenseNet121.
    target_layer = backbone.get_layer("relu")

    if len(target_layer.output.shape) != 4:
        raise RuntimeError("The selected Grad-CAM layer is not spatial.")

    # Connect the nested backbone's intermediate feature map to its output.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
        name="densenet121_gradcam",
    )

    # Reconstruct the outer model's forward path using its existing layers.
    inputs = model.input

    augmentation = model.get_layer("train_augmentation")
    augmented = augmentation(inputs, training=False)

    conv_outputs, features = backbone_grad_model(
        augmented,
        training=False,
    )

    x = features

    # Reuse the exact trained classification-head layers.
    for layer_name in [
        "global_average_pooling",
        "head_batch_norm",
        "head_dropout",
        "head_dense",
        "head_dropout_2",
        "malignant_probability",
    ]:
        layer = model.get_layer(layer_name)

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            x = layer(x, training=False)
        elif isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
        else:
            x = layer(x)

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_outputs, x],
        name="connected_gradcam_model",
    )

    return grad_model


def make_gradcam(raw: bytes) -> tuple[dict, bytes]:
    model = load_model()
    batch = preprocess_image(raw)

    grad_model = _build_gradcam_model(model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            batch,
            training=False,
        )

        # Output neuron 0 represents malignant probability.
        score = predictions[:, 0]

    grads = tape.gradient(score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Gradients are unavailable for the selected Grad-CAM layer."
        )

    # Global average pooling of gradients gives channel importance weights.
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)

    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)[0]

    cam = cam / (
        tf.reduce_max(cam) + tf.keras.backend.epsilon()
    )

    cam = tf.image.resize(
        cam[..., tf.newaxis],
        (IMG_HEIGHT, IMG_WIDTH),
    ).numpy().squeeze()

    heat = np.uint8(np.clip(cam, 0, 1) * 255)

    # Red indicates stronger activation; blue indicates weaker activation.
    heat_rgb = np.stack(
        [heat, np.zeros_like(heat), 255 - heat],
        axis=-1,
    )

    # Create overlay using the original RGB image.
    with Image.open(io.BytesIO(raw)) as image:
        base = image.convert("RGB").resize(
            (IMG_WIDTH, IMG_HEIGHT),
            Image.Resampling.BILINEAR,
        )

    overlay = Image.blend(
        base,
        Image.fromarray(heat_rgb),
        alpha=0.38,
    )

    output = io.BytesIO()
    overlay.save(output, format="PNG")

    # Return the normal inference result alongside the explanation.
    prediction = predict_bytes(raw, "uploaded-image")

    return prediction, output.getvalue()
