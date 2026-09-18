"""T-one REAL VOICE STREET TEST: Greedy vs Beam vs Canonical @10 vs Canonical @15.

ИНТЕРАКТИВНЫЙ РЕЖИМ МИКРОФОНА (--mic)
-------------------------------------
Вы говорите фразы с названиями улиц в микрофон и сразу видите распознавание
ровно четырёх конфигураций декодера на ОДНОЙ записи:

    1. Greedy              — жадное CTC-декодирование (без LM)
    2. Beam                — beam search + KenLM (alpha=0.4, beta=0.9, width=200),
                             БЕЗ hotwords — базовая линия
    3. Canonical @10       — Beam + canonical hotwords из streets.txt,
                             hotword_weight=10
    4. Canonical @15       — Beam + canonical hotwords из streets.txt,
                             hotword_weight=15

Другие веса (3/5/7/20) и дополнительные формы hotwords сознательно НЕ
используются. Акустическая модель запускается РОВНО ОДИН раз на фразу; все
четыре декодера работают на одних и тех же logprobs.

КАК ЗАПУСКАТЬ
-------------
    uv run --group decoders python scripts/test_decoders.py --mic

Дальше по циклу, для каждой фразы:

    1. "Expected street (Enter = неизвестна):" — введите ожидаемую улицу
       ("Лесная", "на Лесной", "улица Сорок лет Победы"). Это ТОЛЬКО оценка
       результата — в декодер она не передаётся. Enter — фраза без оценки.
    2. [ENTER] — начать запись; говорите фразу в микрофон.
    3. [ENTER] (второй) — остановить запись и запустить T-one.
    4. Смотрите блок RESULT: текст каждой конфигурации, найдена ли
       ожидаемая улица (FOUND/NOT FOUND), изменился ли результат относительно
       Beam, помогли hotwords (helped) или навредили (harmed).
    5. [ENTER] — следующая фраза, [q] — завершить. В конце печатается
       SESSION SUMMARY (correct/helped/harmed) — только по фразам, где была
       указана expected street.

НАСТРОЙКА (флаги)
-----------------
    --streets PATH             файл улиц (по умолчанию ./streets.txt)
    --threshold 0.01           RMS-порог VAD для авто-режима записи
    --max-record-seconds 30    предохранитель длины записи в --mic
    --beam-width 200           ширина луча (официальное значение T-one)
    --kenlm-path PATH          локальный kenlm.bin (иначе — HF-артефакт t-tech/T-one)
    --model-path PATH          локальный model.onnx (иначе — HF-артефакт t-tech/T-one)
    --results-dir PATH         базовый каталог результатов (по умолчанию ./results)

Пакетные режимы для перепроверки записей (те же 4 конфигурации):
    uv run --group decoders python scripts/test_decoders.py --audio путь/фраза.wav
    uv run --group decoders python scripts/test_decoders.py --audio-dir путь/каталог/

ЧТО СОХРАНЯЕТСЯ
---------------
Каждый запуск создаёт уникальный каталог results/<YYYY-MM-DD_HH-MM-SS>/ с:
    run.json, hotwords.json, phrase_NNN.wav, phrase_NNN.json,
    results.jsonl, results.csv, session_summary.json

Research-only script: no training, no fine-tuning, no changes to the acoustic
model, KenLM or the hotword implementation. WAV: 8 kHz mono PCM16, как в
production-пайплайне T-one.
"""
from __future__ import annotations

import argparse
import datetime
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t_one_train.decoder_experiment import (  # noqa: E402
    MIC_RESULT_FIELDS,
    SESSION_FIELDS,
    TARGET_RATE,
    append_results_jsonl,
    build_hotwords_json,
    build_mic_config_plan,
    build_mic_phrase_record,
    build_session_summary,
    env_versions,
    evaluate_mic_texts,
    load_wav_any,
    mic_config_dicts,
    mic_config_rows,
    mic_display_label,
    run_dir,
    save_wav,
    session_summary_rows,
    write_json,
    write_rows_csv,
)
from src.t_one_train.hotwords import load_streets  # noqa: E402
from src.t_one_train.street_forms_experiment import (  # noqa: E402
    build_canonical_index,
    build_empty_index,
    load_street_names,
    resolve_expected_street,
)

DEFAULT_STREETS = ROOT / "streets.txt"
MIC_CONFIG_WEIGHTS = (10.0, 15.0)  # canonical@10 / canonical@15 — fixed by the test plan

EXAMPLES = """\
примеры:
  uv run --group decoders python scripts/test_decoders.py --mic
  uv run --group decoders python scripts/test_decoders.py --audio phrase.wav
  uv run --group decoders python scripts/test_decoders.py --audio-dir wavs/
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_decoders",
        description=(
            "T-one REAL VOICE STREET TEST: Greedy vs Beam vs Canonical @10 vs "
            "Canonical @15. Один acoustic forward на фразу, четыре декодера."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true",
                      help="интерактивный микрофонный тест: ENTER — начать запись, ENTER — остановить")
    mode.add_argument("--audio", type=Path, help="один WAV-файл (те же 4 конфигурации)")
    mode.add_argument("--audio-dir", type=Path, help="каталог с WAV-файлами")
    p.add_argument("--streets", type=Path, default=DEFAULT_STREETS,
                   help="путь к streets.txt (по умолчанию ./streets.txt)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results",
                   help="базовый каталог результатов (по умолчанию ./results)")
    p.add_argument("--kenlm-path", type=Path, default=None,
                   help="локальный kenlm.bin (по умолчанию официальный артефакт t-tech/T-one)")
    p.add_argument("--model-path", type=Path, default=None,
                   help="локальный model.onnx (по умолчанию официальный артефакт t-tech/T-one)")
    p.add_argument("--beam-width", type=int, default=200,
                   help="beam width (по умолчанию 200 — официальное значение T-one)")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="RMS-порог VAD рекордера (по умолчанию 0.01)")
    p.add_argument("--max-record-seconds", type=float, default=30.0,
                   help="предохранитель длины записи в --mic (по умолчанию 30 c)")
    return p.parse_args(argv)


def print_banner(args: argparse.Namespace, hotword_count: int, run_path: Path, plan: list) -> None:
    line = "=" * 50
    print(line)
    print("T-one REAL VOICE STREET TEST")
    print(line)
    print("Configurations:")
    print()
    for i, (label, kind, weight) in enumerate(plan, 1):
        extra = f" (hotword_weight={weight:g})" if weight is not None else ""
        print(f"{i}. {mic_display_label(label)}{extra}")
    print()
    print(f"Streets: {args.streets} (canonical hotwords: {hotword_count})")
    print(f"Beam: width={args.beam_width}, alpha=0.4, beta=0.9 (официальные значения T-one)")
    print(f"Run dir: {run_path}")
    print(line)
    print("Инструкция:")
    print("  1. Введите expected street (только для оценки, декодеру не передаётся).")
    print("  2. [ENTER] — начать запись и говорите фразу в микрофон.")
    print("  3. Второй [ENTER] — остановить запись и декодировать 4 конфигурации.")
    print("  4. [ENTER] — следующая запись, [q] — завершить (покажет SESSION SUMMARY).")
    print(line)



class MicSession:
    """Owns model + decoders + street indexes; ONE acoustic forward per phrase."""

    def __init__(
        self,
        args: argparse.Namespace,
        model,
        trio,
        hotwords: list[str],
        street_names: list[str],
        out: Path,
        plan: list[tuple[str, str, float | None]],
    ) -> None:
        self.args = args
        self.model = model
        self.trio = trio
        self.hotwords = hotwords
        self.street_names = street_names
        # canonical-only scoring: no experimental forms are part of this test
        self.canonical_index = build_canonical_index(street_names)
        self.additional_index = build_empty_index("additional")
        self.combined_index = self.canonical_index
        self.out = out
        self.plan = plan
        self.configs = mic_config_dicts(plan, len(hotwords))
        self.records: list[dict] = []
        self.phrase_no = 0

    def process(self, pcm16: np.ndarray, sr: int, source: str, expected) -> None:
        """Save WAV -> acoustic forward ONCE -> decode 4 configs -> score -> save."""
        from src.t_one_train.test_decoders_core import collect_phrase_logprobs

        self.phrase_no += 1
        pid = f"phrase_{self.phrase_no:03d}"
        wav_path = self.out / f"{pid}.wav"
        save_wav(wav_path, pcm16, sr)

        phrase_logprobs, _ = collect_phrase_logprobs(self.model, pcm16.astype(np.int32))
        if not phrase_logprobs:
            print(f"\n[{pid}] T-one не нашёл речи в записи (WAV сохранён: {wav_path.name})")
            write_json(self.out / f"{pid}.json", {
                "id": pid,
                "source": source,
                "audio": wav_path.name,
                "error": "no phrases detected",
                "sample_rate": sr,
                "duration_seconds": round(len(pcm16) / sr, 3),
                "expected_street_input": expected.user_input or None,
                "expected_street": expected.street,
                "evaluated": False,
            })
            return

        # один набор logprobs для всех четырёх конфигураций
        logprobs = np.concatenate(phrase_logprobs, axis=0)
        texts = self.trio.decode_plan(logprobs, self.plan, self.hotwords)
        evaluation = evaluate_mic_texts(
            texts,
            self.plan,
            expected_street=expected.street or "",
            reference=expected.reference,
            canonical_index=self.canonical_index,
            additional_index=self.additional_index,
            combined_index=self.combined_index,
            street_names=self.street_names,
        )
        rec = build_mic_phrase_record(
            phrase_id=pid,
            source=source,
            wav_name=wav_path.name,
            expected_street=expected.street,
            expected_street_input=expected.user_input or None,
            evaluated=bool(expected.user_input),
            configs=self.configs,
            decoded=texts,
            evaluation=evaluation,
            sample_rate=sr,
            duration_seconds=len(pcm16) / sr,
        )
        write_json(self.out / f"{pid}.json", rec)
        self.records.append(rec)
        append_results_jsonl(self.out, rec)
        print_result_block(rec)


def print_result_block(rec: dict) -> None:
    """RESULT block: expected, then every config text + street/comparison verdicts."""
    line = "=" * 50
    evaluated = bool(rec.get("evaluated"))
    expected = rec.get("expected_street") or rec.get("expected_street_input")
    print()
    print(line)
    print(f"RESULT — {rec['id']}")
    print(line)
    print("Expected:")
    print(f"  {expected if expected else '(не указана)'}")
    print("-" * 50)
    baseline = rec["decoded"].get("beam_no_hotwords", "")
    for cfg in rec["configs"]:
        label = cfg["label"]
        ev = rec["evaluation"].get(label) or {}
        print(f"{cfg['display']}:")
        print(f"  {rec['decoded'].get(label, '')}")
        if evaluated:
            print(f"  street: {'FOUND' if ev.get('found') else 'NOT FOUND'}")
        if cfg["kind"] == "canonical":
            print(f"  changed: {'YES' if ev.get('changed_from_beam') else 'NO'}")
            if evaluated:
                if ev.get("helped"):
                    print("  helped: YES")
                elif ev.get("harmed"):
                    print("  helped: NO (harmed: YES)")
                else:
                    print("  helped: NO")
            detected = ev.get("detected_streets") or []
            if detected:
                print(f"  detected: {', '.join(detected)}")
            if ev.get("changed_from_beam"):
                print(f"  beam baseline: {baseline}")
            if ev.get("unsupported_streets"):
                print(f"  unsupported: {', '.join(ev['unsupported_streets'])}")
            if ev.get("requires_manual_review"):
                for reason in ev.get("review_reasons") or []:
                    print(f"  review: {reason}")
    print(line)



def print_session_summary(summary: dict) -> None:
    line = "=" * 50
    print()
    print(line)
    print("SESSION SUMMARY")
    print(line)
    print(f"Tests: {summary['tests']}")
    print(f"Evaluated: {summary['evaluated']}")
    print()
    for row in summary["configs"]:
        print(f"{row['display']}:")
        print(f"  correct: {row['correct']}/{row['evaluated']}")
        if row["kind"] == "canonical":
            print(f"  helped vs Beam: {row['helped']}")
            print(f"  harmed vs Beam: {row['harmed']}")
        print()
    if summary["evaluated"] == 0:
        print("(нет оценённых фраз: expected street не была указана ни разу)")
    print(line)


def record_manual_enter(recorder, max_seconds: float) -> tuple[np.ndarray, int]:
    """Record until the operator presses ENTER again (background stdin watcher)."""
    stop = threading.Event()

    def wait_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        finally:
            stop.set()

    watcher = threading.Thread(target=wait_enter, daemon=True)
    watcher.start()
    result = recorder.record_manual(stop.is_set, max_seconds=max_seconds)
    watcher.join(timeout=1.0)  # no lingering stdin thread at interpreter shutdown
    return result


def run_mic(args: argparse.Namespace, session: MicSession) -> None:
    from src.t_one_train.mic_recorder import MicRecorder, VADConfig

    recorder = MicRecorder(VADConfig(threshold=args.threshold))
    while True:
        print()
        try:
            user_input = input("Expected street (Enter = неизвестна):\n> ").strip()
        except EOFError:
            break
        expected = resolve_expected_street(user_input, session.street_names)
        if user_input and not expected.resolved:
            if expected.candidates:
                print("  ⚠ в streets.txt нет точного совпадения; кандидаты: "
                      + ", ".join(expected.candidates))
            else:
                print("  ⚠ не найдено в streets.txt; оценка будет по введённому тексту")

        try:
            cmd = input("[ENTER] — начать запись, [q] — выйти > ")
        except EOFError:
            break
        if cmd.strip().lower() == "q":
            break

        print("● Запись... говорите. [ENTER] — остановить.", flush=True)
        pcm, _rate = record_manual_enter(recorder, max_seconds=args.max_record_seconds)
        print("■ Запись остановлена. Один acoustic forward, декодирование 4 конфигураций...",
              flush=True)
        if len(pcm) < TARGET_RATE // 10:
            print("  ⚠ запись почти пустая — фраза не распознавалась, повторите.")
            continue
        session.process(pcm, TARGET_RATE, "mic", expected)

        try:
            again = input("[ENTER] — следующая запись, [q] — завершить > ")
        except EOFError:
            break
        if again.strip().lower() == "q":
            break


def finish(session: MicSession, print_summary: bool) -> None:
    """Persist results.csv / session_summary.json + print the summary."""
    if not session.records:
        return
    rows: list[dict] = []
    for rec in session.records:
        rows.extend(mic_config_rows(rec))
    write_rows_csv(session.out / "results.csv", MIC_RESULT_FIELDS, rows)
    summary = build_session_summary(session.records, session.configs)
    write_json(session.out / "session_summary.json", summary)
    write_rows_csv(session.out / "session_summary.csv", SESSION_FIELDS,
                   session_summary_rows(summary))
    if print_summary:
        print_session_summary(summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hotwords, hotword_audit = load_streets(args.streets)
    street_names = load_street_names(args.streets)
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)

    out = run_dir(args.results_dir)
    print_banner(args, len(hotwords), out, plan)
    write_json(out / "hotwords.json", build_hotwords_json(hotword_audit, str(args.streets)))

    run_meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": "mic" if args.mic else ("audio" if args.audio else "audio-dir"),
        "source": (str(args.audio) if args.audio
                   else (str(args.audio_dir) if args.audio_dir else "microphone")),
        "streets_file": str(args.streets),
        "hotword_count": len(hotwords),
        "configurations": [
            {"label": label, "kind": kind, "weight": weight, "display": mic_display_label(label)}
            for label, kind, weight in plan
        ],
        "hotword_format": "lowercase full street name incl. 'улица'/'переулок'; digits expanded to words",
        "expected_street_usage": "scoring only; never passed to decoders",
        "kenlm_path": str(args.kenlm_path) if args.kenlm_path else "hf:t-tech/T-one/kenlm.bin (hf_hub cache)",
        "model_path": str(args.model_path) if args.model_path else "hf:t-tech/T-one/model.onnx (hf_hub cache)",
        "beam_width": args.beam_width,
        "decoder_params": {"alpha": 0.4, "beta": 0.9, "beam_width": args.beam_width},
        "acoustic_forwards_note": (
            "one acoustic model forward per utterance; all configs decode the same logprobs"
        ),
        "mic_vad_threshold": args.threshold if args.mic else None,
        "mic_max_record_seconds": args.max_record_seconds if args.mic else None,
        "env": env_versions(),
    }
    write_json(out / "run.json", run_meta)

    from src.t_one_train.test_decoders_core import (
        DecoderTrio,
        load_acoustic_model,
        load_beam_decoder,
    )
    from tone.decoder import GreedyCTCDecoder

    print("Loading T-one acoustic model (первый запуск скачивает, далее кэш)...", flush=True)
    model = load_acoustic_model(str(args.model_path) if args.model_path else None)
    print("Building KenLM beam decoder (alpha=0.4, beta=0.9, один инстанс на все конфигурации)...",
          flush=True)
    beam = load_beam_decoder(str(args.kenlm_path) if args.kenlm_path else None)
    trio = DecoderTrio(greedy=GreedyCTCDecoder(), beam=beam, beam_width=args.beam_width)
    session = MicSession(args, model, trio, hotwords, street_names, out, plan)

    try:
        if args.mic:
            run_mic(args, session)
        elif args.audio:
            pcm, sr = load_wav_any(args.audio)
            session.process(pcm, sr, str(args.audio), resolve_expected_street("", street_names))
        else:
            wavs = sorted(args.audio_dir.glob("*.wav"))
            if not wavs:
                print(f"Нет .wav файлов в {args.audio_dir}")
                return 1
            for w in wavs:
                pcm, sr = load_wav_any(w)
                session.process(pcm, sr, str(w), resolve_expected_street("", street_names))
    except KeyboardInterrupt:
        print("\nЗавершено пользователем (Ctrl+C).")
    finally:
        finish(session, print_summary=bool(session.records))
        print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
