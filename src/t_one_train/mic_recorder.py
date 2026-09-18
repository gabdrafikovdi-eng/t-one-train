"""Microphone capture with a minimal energy VAD for phrase segmentation.

Mac-friendly: sounddevice (PortAudio), 48 kHz native capture (default input
rate), resampled to 8 kHz mono PCM16 for T-one. RMS-based VAD: phrase starts
when energy exceeds threshold, ends after `silence_ms` of quiet. If the
default input device refuses 48 kHz we retry with the device default rate.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

INPUT_RATE = 48_000
TARGET_RATE = 8_000
BLOCK_MS = 30


@dataclass
class VADConfig:
    threshold: float = 0.01  # RMS threshold; adaptive-calibrated on startup
    preroll_ms: int = 300
    silence_ms: int = 700
    max_phrase_s: float = 15.0
    min_phrase_ms: int = 200


def resample_to_8k(x: np.ndarray, orig_rate: int) -> np.ndarray:
    if orig_rate == TARGET_RATE:
        return x
    from scipy.signal import resample_poly

    g = np.gcd(int(orig_rate), TARGET_RATE)
    return resample_poly(x, TARGET_RATE // g, orig_rate // g).astype(np.float32)


class MicRecorder:
    """Record one phrase; blocking call returns int16 samples at 8 kHz."""

    def __init__(self, config: VADConfig | None = None, input_rate: int = INPUT_RATE) -> None:
        self.cfg = config or VADConfig()
        self.input_rate = input_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            pass
        self._q.put(indata[:, 0].copy())

    def _stream(self, rate: int):
        return sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=int(rate * BLOCK_MS / 1000),
            callback=self._callback,
        )

    def record_phrase(self) -> tuple[np.ndarray, int]:
        """Return (audio_int16_at_8k, original_rate)."""
        try:
            stream = self._stream(self.input_rate)
            rate = self.input_rate
            stream.start()
        except sd.PortAudioError:
            rate = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
            stream = self._stream(rate)
            stream.start()

        cfg = self.cfg
        block = int(rate * BLOCK_MS / 1000)
        preroll_blocks = max(1, int(cfg.preroll_ms / BLOCK_MS))
        silence_blocks = max(1, int(cfg.silence_ms / BLOCK_MS))
        max_blocks = int(cfg.max_phrase_s * 1000 / BLOCK_MS)
        min_blocks = max(1, int(cfg.min_phrase_ms / BLOCK_MS))

        preroll: list[np.ndarray] = []
        captured: list[np.ndarray] = []
        speech_started = False
        quiet = 0
        try:
            while True:
                data = self._q.get()
                rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
                if not speech_started:
                    preroll.append(data)
                    if len(preroll) > preroll_blocks:
                        preroll.pop(0)
                    if rms >= cfg.threshold:
                        speech_started = True
                        captured = list(preroll)
                        quiet = 0
                else:
                    captured.append(data)
                    if rms < cfg.threshold:
                        quiet += 1
                        if quiet >= silence_blocks and len(captured) - quiet >= min_blocks:
                            break
                    else:
                        quiet = 0
                    if len(captured) >= max_blocks:
                        break
        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(captured).astype(np.float32)
        audio8k = resample_to_8k(audio, rate)
        peak = np.max(np.abs(audio8k)) or 1.0
        if peak > 1.0:
            audio8k = audio8k / peak
        pcm16 = np.clip(audio8k * 32767.0, -32768, 32767).astype(np.int16)
        return pcm16, rate

    def record_manual(
        self, should_stop, max_seconds: float = 30.0
    ) -> tuple[np.ndarray, int]:
        """Record from start until ``should_stop()`` is True (second ENTER).

        Same capture path and preprocessing as :meth:`record_phrase` (48 kHz
        native stream -> mono float32 blocks -> 8 kHz resample -> peak-safe
        PCM16); only the stop condition is manual instead of the silence VAD.
        ``should_stop`` is polled between audio blocks; ``max_seconds`` is a
        runaway guard. Returns (audio_int16_at_8k, original_rate).
        """
        try:
            stream = self._stream(self.input_rate)
            rate = self.input_rate
            stream.start()
        except sd.PortAudioError:
            rate = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
            stream = self._stream(rate)
            stream.start()

        captured: list[np.ndarray] = []
        max_blocks = int(max_seconds * 1000 / BLOCK_MS)
        try:
            while len(captured) < max_blocks:
                try:
                    captured.append(self._q.get(timeout=0.05))
                except queue.Empty:
                    if should_stop():
                        break
                    continue
                if should_stop():
                    break
        finally:
            stream.stop()
            stream.close()

        if not captured:
            return np.zeros(0, dtype=np.int16), rate
        audio = np.concatenate(captured).astype(np.float32)
        audio8k = resample_to_8k(audio, rate)
        peak = np.max(np.abs(audio8k)) or 1.0
        if peak > 1.0:
            audio8k = audio8k / peak
        pcm16 = np.clip(audio8k * 32767.0, -32768, 32767).astype(np.int16)
        return pcm16, rate
