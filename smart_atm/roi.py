"""Region of interest (ROI): the area in front of the ATM where the user stands.

Only faces inside the ROI are used for the access decision, so people passing
by outside the booth are ignored. The ROI is selected once and saved to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def overlap_fraction(box: Box, roi: Box) -> float:
    """Fraction of `box`'s area that lies inside `roi` (0.0 to 1.0)."""
    x1, y1 = max(box[0], roi[0]), max(box[1], roi[1])
    x2, y2 = min(box[2], roi[2]), min(box[3], roi[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return inter / area if area else 0.0


def inside_roi(box: Box, roi: Optional[Box], min_overlap: float = 0.6) -> bool:
    """True if at least `min_overlap` of the box is inside the ROI (or no ROI is set)."""
    if roi is None:
        return True
    return overlap_fraction(box, roi) >= min_overlap


def load_roi(path: Path) -> Optional[Box]:
    try:
        data = json.loads(Path(path).read_text())
        return tuple(int(v) for v in data["roi"])  # type: ignore[return-value]
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return None


def save_roi(path: Path, roi: Box) -> None:
    Path(path).write_text(json.dumps({"roi": list(roi)}, indent=2))


def select_roi(frame) -> Optional[Box]:
    """Let the operator drag a rectangle on the frame. Returns None if cancelled."""
    import cv2

    x, y, w, h = cv2.selectROI(
        "Select ROI (drag, then ENTER/SPACE; C to cancel)", frame, showCrosshair=True
    )
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        return None
    return int(x), int(y), int(x + w), int(y + h)
