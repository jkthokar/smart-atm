"""Siren playback with no extra dependencies.

A siren sound is generated once as a WAV file and played with the operating
system's built-in player: winsound on Windows, afplay on macOS, and
aplay/paplay on Linux. If none is available, the terminal bell is used.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np


def make_siren_wav(path: Path, seconds: float = 3.0, rate: int = 22050) -> Path:
    """Write a rising/falling two-tone siren to `path`."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    freq = 800 + 500 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t))  # 800-1300 Hz sweep
    phase = 2 * np.pi * np.cumsum(freq) / rate
    samples = (0.6 * np.sin(phase) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


class Siren:
    def __init__(self):
        self._wav = make_siren_wav(Path(tempfile.gettempdir()) / "smart_atm_siren.wav")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, seconds: float = 10.0) -> None:
        if self.active:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(seconds,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if sys.platform.startswith("win"):
            import winsound

            winsound.PlaySound(None, 0)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self, seconds: float) -> None:
        end = time.time() + seconds
        if sys.platform.startswith("win"):
            import winsound

            winsound.PlaySound(str(self._wav), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            self._stop.wait(seconds)
            winsound.PlaySound(None, 0)
            return

        player = next((p for p in ("afplay", "paplay", "aplay") if shutil.which(p)), None)
        while not self._stop.is_set() and time.time() < end:
            if player:
                self._proc = subprocess.Popen(
                    [player, str(self._wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                while self._proc.poll() is None and not self._stop.is_set() and time.time() < end:
                    time.sleep(0.1)
                if self._proc.poll() is None:
                    self._proc.terminate()
            else:
                print("\a[ALARM] Tampering detected!", flush=True)
                self._stop.wait(1.0)
