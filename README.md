# t-one-train

Локальный synthetic telephone dataset для эксперимента **original T-one vs
fine-tuned T-one**. На Mac обучение НЕ запускается. Внешние ASR datasets
не подключаются: проверяется узкая адаптация pretrained модели к улицам.

## Данные

В `streets.txt` 108 непустых строк, но **107 уникальных улиц**: строка 91 —
заголовок `Страница 2 (улицы 91–107)`. Он исключается с записью в отчёт,
исходный файл не меняется. Согласовано **107 × 500 = 53 500** samples:
48 150 train / 2 675 validation / 2 675 test; каждая улица 450/25/25.
Написание `Целлинная` сохранено без догадок. Улица Искра и переулок Искра
не объединяются в одну сущность.

## Официальный T-one workflow

Изучены 18.09.2026:
- https://huggingface.co/t-tech/T-one
- https://github.com/voicekit-team/T-one (commit `3c5b6c015038173840e62cea99e10cdb1c759116`)
- https://github.com/voicekit-team/T-one/blob/main/examples/finetune_example.ipynb
- https://huggingface.co/t-tech/T-one/blob/main/vocab.json
- https://github.com/voicekit-team/T-one/blob/main/tone/training/model_wrapper.py
- https://github.com/voicekit-team/T-one/blob/main/tone/training/data_collator.py

71M Conformer, CTC; более 80 тысяч часов предварительного обучения,
57,9 тысяч часов телефонной речи. Не требуется повторять предобучение
случайными внешними данными.

Notebook читает JSONL `{"audio": "путь.wav", "text": "текст"}` через
`datasets.load_dataset("json", ...)`, затем `Audio(sampling_rate=8000)`.
Tokenizer: `Wav2Vec2CTCTokenizer.from_pretrained("t-tech/T-one",
pad_token="[PAD]", word_delimiter_token="|")`.
В HF нет `preprocessor_config.json` (404): notebook **создаёт**
`Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=8000,
padding_value=0., return_attention_mask=False, do_normalize=False)`.

Transcript: `" ".join(re.findall("[а-яё]+", text.lower()))`.
**Ё сохраняется** — она есть в vocabulary. Пунктуация и дефисы становятся
границами слов. Числа заранее раскрываются словами: `65 лет Победы` →
`шестьдесят пять лет победы`; иначе regex удалит их.
При обучении добавляется **по 300 ms нулей** слева и справа;
в WAV это training padding не вшито. Используется официальный CTC collator,
labels padding = -100, input_lengths обязательны. Checkpoint:
`ToneForCTC.from_pretrained("t-tech/T-one")`, НЕ случайная инициализация.
WER notebook — greedy CTC decoding, labels декодируются без группировки.

## Silero и лицензии

Источники: https://github.com/snakers4/silero-models и официальный `models.yml`.
Локальная актуальная `v5_5_ru`, веса:
https://models.silero.ai/models/tts/ru/v5_5_ru.pt.
Подтверждены `model.speakers`: aidar, baya, kseniya, xenia, eugene.
SHA-256 весов зафиксирован в коде и provenance. Сначала MPS, при ошибке CPU.
На проверенном Mac M2 модель отклонила MPS при `.to()`; работает CPU,
4 torch threads, один экземпляр модели. LLM освобождается до TTS.

**Silero v5_5_ru: CC BY-NC 4.0.** Уточните права на коммерческое обучение,
использование результатов и распространение у Silero. Этот проект не
обещает разрешения коммерческого внедрения. T-one — Apache 2.0.
MIT CIS-модели Silero имеют другие голоса и требования к ударениям и не
подменяют выбранную модель автоматически.

## Тексты и splits

Ручные семейства + локальная Ollama `qwen2.5:1.5b` Q4_K_M.
LLM предлагает контексты с одним `{pre}/{acc}/{gen}`, **не адреса**.
Консервативная проверяемая грамматика отвергает неизвестные конструкции,
мусор, посторонние названия, повторные адреса, слишком длинные строки.
Не заявляется универсальное определение осмысленности произвольного текста.
Ответы, принятые контексты, provenance — `assets/llm_templates.json`.

Падеж несёт слово улица/переулок. Уже родительные антропонимы (Ленина,
Ахмета Лутфуллина) повторно не склоняются. Прилагательные согласуются:
Весенняя → на Весенней улице / на Весеннюю улицу. Для редких топонимов
используются полные названия с типом улица/переулок.

Splits имеют раздельные базовые семейства **до синтеза**, отдельный банк test.
Приветствия/короткие продолжения общие, основная test-конструкция новая.
LLM-контексты идут только в train (цель 10%; принято 8 контекстов от qwen2.5:1.5b, их недостаточно для большей доли). Все нормализованные тексты
плана уникальны глобально: другой голос не создаёт новый text example.
Вариативность: фразы, голоса, длина, пунктуационные паузы, скорость 0.95–1.05
с сохранением высоты (ffmpeg atempo), разные телефонные профили.

## Телефонный канал

Silero 24 kHz mono → anti-alias resampling 8 kHz → полоса 300–3400 Hz →
громкость/шум/редкий clipping → вероятностный настоящий G.711 A-law или
μ-law encode/decode через ffmpeg → PCM16 WAV mono 8 kHz.
Профили clean/normal/noisy: примерно 30/55/15%. Noise probabilities
0/0.20/0.70; SNR normal 26–34 dB, noisy 18–26 dB. Codec probability 0.5;
volume probability 0.3. Шум синтезированный полосовой, не внешний dataset.
Это приближение канала, не полноценная симуляция сети/микрофона.


## Установка и запуск (Mac M2, 8 GB)

Нужны Python 3.12, uv, ffmpeg, Ollama. Из корня проекта:

```bash
brew install ffmpeg ollama
ollama serve                       # отдельный терминал, 127.0.0.1:11434
ollama pull qwen2.5:1.5b
uv sync --locked
uv run python main.py prepare-texts
uv run python main.py generate --samples-per-street 2
uv run python main.py validate --output dataset_2
uv run python main.py stats --output dataset_2
uv run pytest
```

При 2 samples/street математически невозможно дать каждой улице все три
splits: smoke = 1 train + 1 test на улицу, пустой validation. При 10 — 8/1/1;
при 500 — 450/25/25.

```bash
uv run python main.py generate --samples-per-street 500
uv run python main.py validate
uv run python main.py stats
uv run pytest
```

Другие размеры: 100, 300, 1000; отдельный output выбирается автоматически
(`dataset_<N>`), `--output` переопределяет. Все параметры — `config.toml`.
Генерация строго последовательная; параллельные TTS workers на 8 GB не нужны.

## Resume и целостность

Повторите ту же команду. ID детерминированы, план immutable; изменение config,
исходного кода, списка улиц или LLM-cache запрещает смешивание запусков
(сохраняются SHA-256 исходников и fingerprint). Каждый WAV имеет атомарный
sidecar с хешем декодированных samples. Корректные WAV не синтезируются
снова; отсутствующие/повреждённые восстанавливаются. Второй писатель
блокируется flock. Manifest/metadata.jsonl экспортируются атомарно после
всех WAV; после генерации автоматически выполняется validate. Ошибки —
`errors.jsonl`; процесс останавливается, тихо неполного dataset не бывает.

## Проверки

validate читает каждый WAV: PCM16 WAV, 8 kHz, mono, 0.35–19 seconds, finite,
non-silent, hash, counts, manifests, duplicate text/audio, leakage по
text/audio/family. Полная статистика по каждой улице — `report.json`;
`stats` показывает ранее рассчитанный отчёт. В RAM нет всего аудио —
только один sample и компактные индексы для контроля дубликатов.

## Ограничения и следующий эксперимент

Автоматическая проверка не подтверждает ударения и фонетику башкирских
названий: до обучения прослушайте выборку всех улиц и голосов; при ошибках
нужен проверенный pronunciation словарь, а не догадки агента. Авто-ударения
Silero могут отличаться от исходного написания — сверяйте с транскриптом.
Произношение нельзя оценивать только тем ASR, который адаптируем.
Синтетический held-out test измеряет перенос на новые конструкции внутри
Silero-домена, а не гарантированное качество на живых звонках.

A original vs B fine-tuned сравнивать одним decoder и одним test, без
подстройки на test. Метрики: WER, CER; Street Accuracy — доля примеров,
где извлечён корректный canonical street ID по словарю всех допустимых форм;
Exact Street Accuracy — точное полное название, включая улица/переулок.
Не путать с Exact Utterance Accuracy. Для выводов о production нужен
отдельный лицензированный реальный контрольный набор. Внешний тренировочный
ASR dataset рассматривать только после A/B. Fine-tuning запускается на
отдельной машине: RTX 4060 8 GB, 16 GB RAM; на Mac обучение не выполняется.

## Перенос на PC и fine-tuning (RTX 4060 8 GB, 16 GB RAM)

На Mac обучение не выполняется. Каталог dataset копируется целиком
(WAV + train/validation/test.jsonl + metadata.jsonl + report.json + run.json),
например через rsync -a на внешний диск:

```bash
rsync -a --info=progress2 dataset/ /Volumes/USB/tone-dataset/
# на PC:
rsync -a --info=progress2 /mnt/usb/tone-dataset/ ./dataset/
```

На PC (Linux + CUDA 12.x) создайте venv и поставьте официальные зависимости
T-one (см. pyproject.toml этого проекта — секция finetune-зависимостей):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
# CPU-only machine or CUDA base image: install CUDA torch first
pip install "torch>=2.7" "torchaudio>=2.7" --index-url https://download.pytorch.org/whl/cu126
# Official T-one + finetune extra (see pyproject.toml [project.optional-dependencies].finetune)
pip install "tone[finetune] @ git+https://github.com/voicekit-team/T-one"
# street_metrics only needs jiwer (usually pulled in by the extra above)
pip install jiwer
```

Готовый скрипт этого проекта — `train_tone.py` — использует те же параметры
preprocessing, что официальный notebook (8 kHz, `[а-яё]+`, 300 ms паддинг,
`do_normalize=False`, `ToneForCTC.from_pretrained("t-tech/T-one")`,
`DataCollatorCTCWithPadding`). На RTX 4060 8 GB безопасный старт —
`--batch 4 --grad-accum 16` (эффективный batch 64) и `bf16=True`;
если OOM — уменьшайте `--batch`, увеличивая `--grad-accum`.

```bash
# A: original T-one на нашем test (без обучения) + предсказания для метрик улиц
python train_tone.py --eval-only dataset/test.jsonl --model t-tech/T-one \
  --save-predictions preds_A.jsonl

# B: fine-tuning, затем та же метрика на том же test
python train_tone.py --train dataset/train.jsonl \
  --validation dataset/validation.jsonl --batch 4 --grad-accum 16 --epochs 10

python train_tone.py --eval-only dataset/test.jsonl \
  --model tone_ctc/checkpoint-<N> --save-predictions preds_B.jsonl
```

Метрики улиц (WER/CER + Street Accuracy + Exact Street Accuracy) считаются
одним и тем же словарём допустимых форм названий для A и B:

```bash
python -m src.t_one_train.street_metrics --test dataset/test.jsonl --predictions preds_A.jsonl
python -m src.t_one_train.street_metrics --test dataset/test.jsonl --predictions preds_B.jsonl
```

Гипотеза подтверждается, только если B значительно улучшает Street Accuracy
относительно A при том же decoder и том же test. Только после сравнения A/B
решать вопрос о внешнем лицензированном телефонном ASR dataset; на первом
эксперименте внешние данные (Common Voice / Golos / SOVA / OpenSTT) не
подключаются намеренно.
