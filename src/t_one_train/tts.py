"""Local Silero package loader; never loads an ASR training model."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
VOICES = ("aidar", "baya", "kseniya", "xenia", "eugene")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class Silero:
    def __init__(self, device="auto", threads=4):
        torch.set_num_threads(threads)
        self.path = ROOT / ".cache" / "v5_5_ru.pt"
        self.path.parent.mkdir(exist_ok=True)
        if not self.path.exists():
            tmp = self.path.with_suffix(".download")
            print(f"Downloading {MODEL_URL}", flush=True)
            urllib.request.urlretrieve(MODEL_URL, tmp)
            os.replace(tmp, self.path)
        digest = sha256(self.path)
        expected = "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"
        if digest != expected:
            raise ValueError(f"Silero weight integrity mismatch: {digest}")
        self.model = torch.package.PackageImporter(str(self.path)).load_pickle("tts_models", "model")
        self.device = "cpu"
        self.fallback = None
        available = list(self.model.speakers)
        if not set(VOICES) <= set(available):
            raise RuntimeError(f"Unexpected Silero voices: {available}")
        self.model.to(torch.device("cpu"))
        if device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
            try:
                self.model.to(torch.device("mps"))
                self.device = "mps"
                for voice in VOICES:
                    self.speak("Мне нужна машина, пожалуйста.", voice, 123, fallback=False)
            except Exception as e:
                self.fallback = f"MPS probe: {type(e).__name__}: {e}"
                print(self.fallback, flush=True)
                self.model.to(torch.device("cpu"))
                self.device = "cpu"
                torch.mps.empty_cache()
        print(f"Silero device={self.device}, voices={available}", flush=True)

    def speak(self, text, voice, seed, fallback=True):
        torch.manual_seed(seed % (2**32))
        try:
            with torch.inference_mode():
                audio = self.model.apply_tts(text=text, speaker=voice, sample_rate=24000)
            result = audio.detach().cpu().numpy().astype(np.float32)
            if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
                raise ValueError("Invalid Silero output")
            return result
        except Exception as e:
            if self.device != "mps" or not fallback:
                raise
            self.fallback = f"MPS runtime: {type(e).__name__}: {e}"
            print(self.fallback, flush=True)
            self.model.to(torch.device("cpu"))
            self.device = "cpu"
            torch.mps.empty_cache()
            return self.speak(text, voice, seed, fallback=False)

    def provenance(self):
        return dict(url=MODEL_URL, sha256=sha256(self.path), voices=list(VOICES),
                    device=self.device, fallback=self.fallback, torch=torch.__version__)
