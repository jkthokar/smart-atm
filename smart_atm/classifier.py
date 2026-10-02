"""Face-occlusion classifier: is the face visible (grant) or covered (deny)?

The model is MobileNetV2 (ImageNet weights) with a small binary head, trained
by `train.py`. Pre-processing is built into the model, so it takes raw RGB
images of size `img_size` x `img_size` with pixel values 0-255.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def build_model(img_size: int = 224, dropout: float = 0.3, weights: str | None = "imagenet"):
    """MobileNetV2 backbone + augmentation + binary head. Output: P(access_granted)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_granted")(x)
    return keras.Model(inputs, outputs, name="smart_atm_classifier"), base


class OcclusionClassifier:
    def __init__(self, model_path: Path, img_size: int = 224):
        from tensorflow import keras

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Classifier not found at {model_path}. Train it with `python train.py`.")
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size

    def _prepare(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.img_size, self.img_size)).astype("float32")

    def predict(self, faces_bgr: List[np.ndarray]) -> List[float]:
        """Return P(access_granted) for each face crop (BGR, any size)."""
        if not faces_bgr:
            return []
        batch = np.stack([self._prepare(f) for f in faces_bgr])
        return [float(p) for p in self.model.predict(batch, verbose=0).ravel()]
