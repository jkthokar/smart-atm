"""Run Smart ATM on a webcam or a video file.

    python run.py                      # webcam 0
    python run.py --source demo.mp4    # video file
    python run.py --reset-roi          # choose a new ROI
    python run.py --detector-only      # demo without a trained model

Keys:  Q quit   R reselect ROI   T test the alarm   S silence the alarm
External sensors can trigger the alarm with:  POST http://<this-pc>:8765/tamper
"""

import argparse
import time

import cv2

from smart_atm.alarm import Siren
from smart_atm.classifier import OcclusionClassifier
from smart_atm.config import Config
from smart_atm.decision import AccessController, AccessState, FrameResult
from smart_atm.face_detector import FaceDetector, crop_face
from smart_atm.roi import inside_roi, load_roi, save_roi, select_roi
from smart_atm.tamper import TamperListener

GREEN, RED, AMBER, GREY, WHITE = (60, 200, 60), (40, 40, 230), (0, 190, 255), (160, 160, 160), (255, 255, 255)
STATE_STYLE = {
    AccessState.IDLE: ("Please look at the camera", GREY),
    AccessState.CHECKING: ("Checking...", AMBER),
    AccessState.GRANTED: ("ACCESS GRANTED - insert your card", GREEN),
    AccessState.DENIED: ("ACCESS DENIED - please uncover your face", RED),
}


def draw_banner(frame, text, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 44), (w, h), color, -1)
    cv2.putText(frame, text, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--reset-roi", action="store_true", help="select a new ROI on start")
    ap.add_argument("--no-roi", action="store_true", help="use the whole frame")
    ap.add_argument("--no-tamper-server", action="store_true", help="do not start the HTTP tamper listener")
    ap.add_argument("--mute", action="store_true", help="show alarms on screen without sound")
    ap.add_argument("--detector-only", action="store_true",
                    help="demo without a trained classifier: face detection, ROI and alarm only")
    args = ap.parse_args()

    cfg = Config()
    detector = FaceDetector(cfg.face_prototxt, cfg.face_weights, cfg.face_confidence)
    classifier = None
    if not args.detector_only:
        try:
            classifier = OcclusionClassifier(cfg.classifier_path, cfg.img_size)
        except FileNotFoundError:
            print("No trained classifier found, so running in detector-only demo mode.\n"
                  "Train one with `python train.py` to enable access control.")
    controller = AccessController(cfg.frames_to_grant, cfg.frames_to_deny)
    siren = Siren()

    listener = None
    if not args.no_tamper_server:
        listener = TamperListener(port=cfg.tamper_port).start()
        print(f"Tamper listener on port {listener.port} (POST /tamper)")

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source {args.source!r}")

    roi = None
    if not args.no_roi:
        roi = None if args.reset_roi else load_roi(cfg.roi_file)
        if roi is None:
            roi = select_roi(frame)
            if roi:
                save_roi(cfg.roi_file, roi)

    alarm_until, fps, t_prev = 0.0, 0.0, time.time()
    while ok:
        faces = [(b, c) for b, c in detector.detect(frame) if inside_roi(b, roi, cfg.roi_min_overlap)]
        crops = [crop_face(frame, b, cfg.face_margin) for b, _ in faces]
        probs = classifier.predict(crops) if classifier else []

        view = frame.copy()
        if roi:
            cv2.rectangle(view, roi[:2], roi[2:], WHITE, 1)
            cv2.putText(view, "ROI", (roi[0] + 4, roi[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        if classifier is None:
            for box, conf in faces:
                cv2.rectangle(view, box[:2], box[2:], AMBER, 2)
                cv2.putText(view, f"Face {conf:.0%}", (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2)

        visible_flags = []
        for (box, _), p in zip(faces, probs):
            visible = p >= cfg.grant_threshold
            visible_flags.append(visible)
            label = f"Visible {p:.0%}" if visible else f"Covered {1 - p:.0%}"
            color = GREEN if visible else RED
            cv2.rectangle(view, box[:2], box[2:], color, 2)
            cv2.putText(view, label, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Every face in the ROI must be visible; one covered face denies access.
        if not visible_flags:
            result = FrameResult.NO_FACE
        elif all(visible_flags):
            result = FrameResult.VISIBLE
        else:
            result = FrameResult.COVERED
        state = controller.update(result)

        # Tamper events: external sensors or the T key.
        key = cv2.waitKey(1) & 0xFF
        events = listener.poll() if listener else []
        if key == ord("t"):
            events.append({"source": "keyboard"})
        if events:
            print(f"[ALARM] tamper event(s): {events}")
            alarm_until = time.time() + cfg.alarm_seconds
            if not args.mute:
                siren.start(cfg.alarm_seconds)
        if key == ord("s"):
            alarm_until = 0.0
            siren.stop()

        text, color = STATE_STYLE[state]
        if classifier is None:
            text, color = "DEMO MODE: face detection only", GREY
        if time.time() < alarm_until:
            text, color = "ALARM: TAMPERING DETECTED", RED
        draw_banner(view, text, color)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(view, f"{fps:4.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2)
        cv2.imshow("Smart ATM", view)

        if key == ord("q"):
            break
        if key == ord("r") and not args.no_roi:
            new_roi = select_roi(frame)
            if new_roi:
                roi = new_roi
                save_roi(cfg.roi_file, roi)
                controller.reset()

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    siren.stop()
    if listener:
        listener.stop()


if __name__ == "__main__":
    main()
