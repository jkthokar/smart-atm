"""Build the dataset from your webcam.

    python collect_data.py --person jay     (use a name without "_")

Hold a key to save face crops into a class folder:
    G  -> dataset/raw/access_granted   (face clearly visible)
    D  -> dataset/raw/access_denied    (mask, helmet, cap, hood, scarf ...)
    Q  -> quit

Only faces the detector can find are saved. This matches how the live system
works: if the detector cannot find a face, access is denied anyway.
Tip: vary lighting, distance, angle, glasses and people for a better model.
"""

import argparse
import time
from pathlib import Path

import cv2

from smart_atm.config import CLASS_NAMES, ROOT, Config
from smart_atm.face_detector import FaceDetector, crop_face

KEYS = {ord("d"): CLASS_NAMES[0], ord("g"): CLASS_NAMES[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "raw")
    ap.add_argument("--person", default="person1",
                    help="name of the person being recorded (used by split_dataset.py --by-person)")
    ap.add_argument("--interval", type=float, default=0.25, help="seconds between saved crops")
    args = ap.parse_args()

    cfg = Config()
    detector = FaceDetector(cfg.face_prototxt, cfg.face_weights, cfg.face_confidence)
    for c in CLASS_NAMES:
        (args.out / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    last_save, counts = 0.0, {c: len(list((args.out / c).glob("*.jpg"))) for c in CLASS_NAMES}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detector.detect(frame)
        view = frame.copy()
        for box, conf in faces:
            cv2.rectangle(view, box[:2], box[2:], (255, 255, 0), 2)
        info = f"granted: {counts[CLASS_NAMES[1]]}   denied: {counts[CLASS_NAMES[0]]}   (G/D to save, Q to quit)"
        cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Collect data", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS and faces and time.time() - last_save >= args.interval:
            label = KEYS[key]
            box, _ = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
            crop = crop_face(frame, box, cfg.face_margin)
            name = args.out / label / f"{args.person}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(name), crop)
            counts[label] += 1
            last_save = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("Saved images:", counts)


if __name__ == "__main__":
    main()
