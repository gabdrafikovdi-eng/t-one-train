"""Bounded telephone channel, actual G.711 round trip through ffmpeg."""
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

SR = 8000


def telephone(audio, speed, profile, cfg, seed):
    rng = np.random.default_rng(seed)
    # Resample with anti-aliasing; preserve pitch during modest speed variation.
    x = resample_poly(audio, 1, 3).astype(np.float32)
    if abs(speed-1) > 1e-6:
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "8000",
            "-ac", "1", "-i", "pipe:0", "-af", f"atempo={speed}",
            "-f", "f32le", "pipe:1"], input=x.tobytes(), capture_output=True, check=True)
        x = np.frombuffer(p.stdout, dtype="<f4").copy()
    x = sosfilt(butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos"), x)
    rms = float(np.sqrt(np.mean(x*x)))
    if rms < 1e-5:
        raise ValueError("Silent TTS output")
    x *= min(0.09/rms, 0.88/max(np.max(np.abs(x)), 1e-8))
    gain_db = float(rng.uniform(-4, 2)) if rng.random() < cfg["augmentation"]["volume_probability"] else 0.
    x *= 10**(gain_db/20)
    p = cfg["profiles"][profile]
    snr = None
    if rng.random() < p["noise_probability"]:
        snr = float(rng.uniform(p["snr_min"], p["snr_max"]))
        noise = sosfilt(butter(2, [300, 3400], btype="bandpass", fs=SR, output="sos"), rng.normal(size=len(x)))
        noise *= np.sqrt(np.mean(x*x)) / (10**(snr/20) * np.sqrt(np.mean(noise*noise)))
        x += noise
    clipped = rng.random() < p["clip_probability"]
    if clipped:
        limit = np.quantile(np.abs(x), .999)
        x = np.clip(x, -limit, limit)
    x = np.clip(x, -.99, .99)
    codec = "none"
    if rng.random() < cfg["augmentation"]["codec_probability"]:
        codec = str(rng.choice(["pcm_mulaw", "pcm_alaw"]))
        raw = np.round(x*32767).astype("<i2").tobytes()
        encoded = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "8000", "-ac", "1",
            "-i", "pipe:0", "-c:a", codec, "-f", "wav", "pipe:1"], input=raw, capture_output=True, check=True)
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ar", "8000",
            "-ac", "1", "pipe:1"], input=encoded.stdout, capture_output=True, check=True)
        x = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    # Short varying channel-edge pauses, NOT the training notebook's 300ms padding.
    pauses = rng.integers(160, 801, size=2)
    x = np.pad(x, tuple(pauses)).astype(np.float32)
    return x, dict(codec=codec, snr_db=snr, gain_db=gain_db, clipped=bool(clipped),
                   edge_pause_ms=(pauses/8).tolist())


def inspect_audio(path):
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 1 or info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError(f"Wrong WAV format: {path}: {info}")
    x, sr = sf.read(path, dtype="float32")
    if not .35 <= len(x)/sr <= 19.0:
        raise ValueError(f"Duration outside [.35,19] seconds: {path}")
    if not np.isfinite(x).all() or np.max(np.abs(x)) < .005 or np.sqrt(np.mean(x*x)) < .0005:
        raise ValueError(f"Invalid or silent signal: {path}")
    return dict(duration=len(x)/sr, audio_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                bytes=Path(path).stat().st_size)
