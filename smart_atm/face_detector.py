"""Face detection with OpenCV's DNN module (ResNet-10 SSD, 300x300 input).

This detector runs in real time on a normal CPU, so no GPU is needed at the
ATM. Run `python download_models.py` once to fetch the model files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .roi import Box

Detection = Tuple[Box, float]  # (box, confidence)


class FaceDetector:
    def __init__(self, prototxt: Path, weights: Path, confidence: float = 0.5):
        if not Path(prototxt).exists() or not Path(weights).exists():
            raise FileNotFoundError(
                "Face detector files not found. Run `python download_models.py` first."
            )
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return face boxes (x1, y1, x2, y2) in pixel coordinates with confidences."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        out = self.net.forward()  # shape (1, 1, N, 7)

        faces: List[Detection] = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < self.confidence:
                continue
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                faces.append(((int(x1), int(y1), int(x2), int(y2)), conf))
        return faces


def crop_face(frame: np.ndarray, box: Box, margin: float = 0.15) -> np.ndarray:
    """Crop a face with an extra margin so caps, hoods and scarves stay in view."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return frame[max(0, y1 - my): min(h, y2 + my), max(0, x1 - mx): min(w, x2 + mx)]
