"""Central configuration. Edit values here instead of hard-coding them in scripts."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Class folder names. Keras orders classes alphabetically, so index 0 is
# "access_denied" and index 1 is "access_granted"; the classifier's sigmoid
# output is therefore P(access_granted).
CLASS_NAMES = ("access_denied", "access_granted")


@dataclass
class Config:
    # --- Face detector (OpenCV DNN, ResNet-10 SSD) ---
    face_prototxt: Path = MODELS_DIR / "deploy.prototxt"
    face_weights: Path = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    face_confidence: float = 0.5          # minimum detector confidence
    face_margin: float = 0.15             # extra border around each face crop

    # --- Occlusion classifier ---
    classifier_path: Path = MODELS_DIR / "smart_atm_classifier.keras"
    img_size: int = 224
    grant_threshold: float = 0.80         # P(granted) needed to count a frame as "visible face"

    # --- Access decision ---
    frames_to_grant: int = 8              # consecutive visible-face frames before granting
    frames_to_deny: int = 5               # consecutive covered/no-face frames before denying
    roi_min_overlap: float = 0.6          # fraction of a face box that must lie inside the ROI

    # --- Tamper alarm ---
    tamper_port: int = 8765               # HTTP port for external sensors (phone / ESP32)
    alarm_seconds: float = 10.0           # siren duration per tamper event

    roi_file: Path = ROOT / "roi.json"
