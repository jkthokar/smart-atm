"""Access decision logic.

The controller is deliberately *fail-safe*: access is granted only after a
visible face has been confirmed for several consecutive frames. A covered face
or a person whose face the detector cannot find leads to denial, never to
access. Requiring consecutive frames also stops the decision from flickering
because of a single noisy prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AccessState(str, Enum):
    IDLE = "IDLE"            # nobody in front of the ATM
    CHECKING = "CHECKING"    # someone present, decision not yet stable
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class FrameResult(str, Enum):
    VISIBLE = "visible"      # face found and classified as uncovered
    COVERED = "covered"      # face found and classified as covered
    NO_FACE = "no_face"      # nobody (or no detectable face) in the ROI


@dataclass
class AccessController:
    frames_to_grant: int = 8
    frames_to_deny: int = 5
    frames_to_idle: int = 15

    state: AccessState = AccessState.IDLE
    _visible: int = 0
    _covered: int = 0
    _empty: int = 0

    def update(self, result: FrameResult, person_present: Optional[bool] = None) -> AccessState:
        """Feed one frame's result and return the current state.

        `person_present` lets the caller say that someone is in the booth even
        though no face was detected (for example, from motion in the ROI). A
        present person with no detectable face is treated like a covered face.
        """
        if result is FrameResult.NO_FACE and person_present:
            result = FrameResult.COVERED

        if result is FrameResult.VISIBLE:
            self._visible += 1
            self._covered = 0
            self._empty = 0
        elif result is FrameResult.COVERED:
            self._covered += 1
            self._visible = 0
            self._empty = 0
        else:
            self._empty += 1
            self._visible = 0
            self._covered = 0

        if self._covered >= self.frames_to_deny:
            self.state = AccessState.DENIED
        elif self._visible >= self.frames_to_grant:
            self.state = AccessState.GRANTED
        elif self._empty >= self.frames_to_idle:
            self.state = AccessState.IDLE
        elif result is not FrameResult.NO_FACE and self.state is AccessState.IDLE:
            self.state = AccessState.CHECKING
        # A granted session is revoked as soon as covering is confirmed (above);
        # otherwise the previous state is kept.
        return self.state

    def reset(self) -> None:
        self.state = AccessState.IDLE
        self._visible = self._covered = self._empty = 0
