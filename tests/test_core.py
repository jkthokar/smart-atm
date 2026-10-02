import json
import urllib.request

import pytest

from smart_atm.decision import AccessController, AccessState, FrameResult
from smart_atm.roi import inside_roi, load_roi, overlap_fraction, save_roi
from smart_atm.tamper import TamperListener

V, C, N = FrameResult.VISIBLE, FrameResult.COVERED, FrameResult.NO_FACE


def feed(ctrl, results):
    for r in results:
        state = ctrl.update(r)
    return state


# ---- ROI ---------------------------------------------------------------
def test_overlap_full_partial_none():
    roi = (0, 0, 100, 100)
    assert overlap_fraction((10, 10, 50, 50), roi) == 1.0
    assert overlap_fraction((50, 0, 150, 100), roi) == pytest.approx(0.5)
    assert overlap_fraction((200, 200, 250, 250), roi) == 0.0


def test_inside_roi_threshold_and_no_roi():
    roi = (0, 0, 100, 100)
    assert inside_roi((50, 0, 150, 100), roi, min_overlap=0.5)
    assert not inside_roi((50, 0, 150, 100), roi, min_overlap=0.6)
    assert inside_roi((500, 500, 600, 600), None)


def test_roi_roundtrip(tmp_path):
    path = tmp_path / "roi.json"
    assert load_roi(path) is None
    save_roi(path, (1, 2, 3, 4))
    assert load_roi(path) == (1, 2, 3, 4)


# ---- Access decision ---------------------------------------------------
def test_grant_requires_consecutive_visible_frames():
    ctrl = AccessController(frames_to_grant=5, frames_to_deny=3)
    assert feed(ctrl, [V] * 4) == AccessState.CHECKING
    assert feed(ctrl, [V]) == AccessState.GRANTED


def test_single_noisy_frame_does_not_grant():
    ctrl = AccessController(frames_to_grant=5, frames_to_deny=3)
    assert feed(ctrl, [V, V, V, C, V, V]) != AccessState.GRANTED


def test_covered_face_denied_and_revokes_grant():
    ctrl = AccessController(frames_to_grant=3, frames_to_deny=3)
    assert feed(ctrl, [V] * 3) == AccessState.GRANTED
    assert feed(ctrl, [C] * 3) == AccessState.DENIED


def test_no_face_never_grants_and_returns_to_idle():
    ctrl = AccessController(frames_to_grant=3, frames_to_deny=3, frames_to_idle=5)
    assert feed(ctrl, [N] * 100) == AccessState.IDLE
    feed(ctrl, [C] * 3)
    assert feed(ctrl, [N] * 5) == AccessState.IDLE


def test_person_present_without_face_is_denied():
    ctrl = AccessController(frames_to_grant=3, frames_to_deny=3)
    for _ in range(3):
        state = ctrl.update(N, person_present=True)
    assert state == AccessState.DENIED


# ---- Tamper listener ---------------------------------------------------
def test_tamper_listener_receives_post():
    listener = TamperListener(host="127.0.0.1", port=0).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{listener.port}/tamper",
            data=json.dumps({"sensor": "test"}).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        events = listener.poll()
        assert len(events) == 1 and events[0]["sensor"] == "test"
        assert listener.poll() == []
    finally:
        listener.stop()
