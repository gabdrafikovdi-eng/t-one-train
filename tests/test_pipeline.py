"""Offline invariants plus optional real Silero end-to-end diagnostic."""
import json
import os
import tomllib
from pathlib import Path
import pytest
import soundfile as sf
from src.t_one_train.forms import forms, normalize, read_streets, render
from src.t_one_train.audio import telephone, inspect_audio
from src.t_one_train.tts import ROOT, VOICES, Silero


def test_streets_and_forms():
    streets, excluded = read_streets(ROOT / "streets.txt")
    assert len(streets) == 107 and len(excluded) == 1
    assert forms("Весенняя улица")["pre"] == "весенней улице"
    assert forms("Весенняя улица")["acc"] == "весеннюю улицу"
    assert forms("Улица Ленина")["pre"] == "улице ленина"
    assert forms("Переулок Искра")["pre"] == "переулке искра"
    assert forms("Улица 65 лет Победы")["nom"] == "улица шестьдесят пять лет победы"
    assert normalize("  Ёлка, Ленина-улица! ") == "ёлка ленина улица"
    for street in streets:
        for case in ("nom", "acc", "pre", "gen"):
            assert forms(street)[case] == normalize(forms(street)[case])


@pytest.mark.skipif(os.getenv("SILERO_INTEGRATION") != "1", reason="Set SILERO_INTEGRATION=1 for real local TTS")
def test_two_streets_end_to_end():
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    streets, _ = read_streets(ROOT / "streets.txt")
    out = ROOT / "diagnostics" / "two_streets_e2e"
    out.mkdir(parents=True, exist_ok=True)
    model = Silero(threads=cfg["threads"])
    rows = []
    profiles = list(cfg["profiles"])
    templates = ["заберите меня на {pre}", "мне нужно на {acc}", "мне нужна машина до {gen}", "{nom}", "я сейчас нахожусь на {pre}"]
    for i, street in enumerate(streets[:2]):
        for j, voice in enumerate(VOICES):
            spoken = render(templates[j], street)
            raw = model.speak(spoken + ".", voice, 100+i*5+j)
            profile = profiles[(i*5+j) % 3]
            speed = [.95, 1., 1.05, .98, 1.02][j]
            audio, aug = telephone(raw, speed, profile, cfg, 100+i*5+j)
            path = out / f"{i:02d}_{voice}.wav"
            sf.write(path, audio, 8000, subtype="PCM_16")
            rows.append(dict(audio=str(path), text=normalize(spoken), street=street,
                             voice=voice, profile=profile, speed=speed, **aug, **inspect_audio(path)))
    (out / "manifest.jsonl").write_text("".join(json.dumps({k:r[k] for k in ('audio','text')}, ensure_ascii=False)+"\n" for r in rows))
    (out / "metadata.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    loaded = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(loaded) == 10
    for row in loaded:
        assert set(row) == {"audio", "text"}
        inspect_audio(row["audio"])
    assert len({r['audio_sha256'] for r in rows}) == 10
    assert {r['voice'] for r in rows} == set(VOICES)
    assert {r['profile'] for r in rows} == set(profiles)
    report = dict(status="passed", samples=10, streets=2, invalid=0,
                  codecs=sorted({r['codec'] for r in rows}),
                  total_duration_seconds=sum(r['duration'] for r in rows),
                  audio_bytes=sum(r['bytes'] for r in rows), provenance=model.provenance())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
