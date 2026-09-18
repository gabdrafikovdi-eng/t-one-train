"""Deterministic per-street configuration."""

from __future__ import annotations

import hashlib
import json


def config_for_street(street: str, *, base_seed: int = 20260918) -> dict:
    """Build the deterministic generation config for one street.

    The same street name always maps to the same voice order, seed and
    profile weights, so re-running generation reproduces the same plan.
    """
    digest = hashlib.sha256(f"{base_seed}:{street}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    # Voice order: rotate the canonical list by the street-dependent offset so
    # different streets lead with different voices while order stays stable.
    voices = ["kseniya", "xenia", "aidar", "baya", "eugene"]
    offset = digest[8] % len(voices)
    ordered = voices[offset:] + voices[:offset]
    # Speed range varies slightly per street: 0.95-1.02 up to 1.03-1.10.
    lo = 0.95 + (digest[9] % 6) * 0.015
    hi = min(lo + 0.08, 1.10)
    # Profile mix per street.
    clean = 0.20 + (digest[10] % 11) * 0.01  # 0.20-0.30
    normal = 0.45 + (digest[11] % 11) * 0.01  # 0.45-0.55
    total = clean + normal
    clean = clean / total * 0.75
    normal = normal / total * 0.75
    return {
        "street": street,
        "seed": seed,
        "voices": ordered,
        "speed_lo": round(lo, 3),
        "speed_hi": round(hi, 3),
        "profiles": {
            "clean_phone": round(clean, 4),
            "normal_phone": round(normal, 4),
            "noisy_phone": round(1.0 - clean - normal, 4),
        },
    }


def config_sha256(cfg: dict) -> str:
    """Stable hash of the per-street config (used in deterministic IDs)."""
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
