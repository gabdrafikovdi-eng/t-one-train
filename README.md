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

## Независимый real-world evaluation set (ручная запись)

Отдельный набор для **объективной оценки** ASR на реальной речи (телефонные звонки
такси, локальные названия улиц). Он независим от датасета генерации:

- **не** используется для fine-tuning T-one;
- **не** является источником hotwords — hotwords строятся только из `streets.txt`;
- один и тот же набор WAV + reference переиспользуется между экспериментами
  (изменение набора ломает сравнимость).

```
streets.txt ──┬──→ hotwords ──→ decoder
              └──→ reference для ручных записей ──→ evaluation set
```

### Запись

```bash
uv run python scripts/record_real_dataset.py            # продолжить последний / создать новый
uv run python scripts/record_real_dataset.py --new      # принудительно новый dataset
uv run python scripts/record_real_dataset.py --dataset results/real_dataset_XXX/
uv run python scripts/record_real_dataset.py --limit 2  # smoke: только 2 улицы
uv run python scripts/validate_real_dataset.py          # последний dataset
uv run python scripts/validate_real_dataset.py results/real_dataset_XXX/
```

Запись требует `sounddevice` (группа `decoders`): если модуль отсутствует, скрипт
попросит `uv sync --group decoders` и предложит запуск через
`uv run --group decoders python scripts/record_real_dataset.py`. Валидация и
парсинг `streets.txt` от микрофона не зависят. `--limit N` создаёт набор только
для первых N улиц — не смешивайте такой smoke-набор с полным прогоном
(`dataset.json` фиксирует план, поэтому валидация сообщит о записях вне плана).

Порядок работы: для каждой улицы из `streets.txt` (в файловом порядке) —
три варианта записи, в каждом: показать reference → ENTER старт записи →
говорите → ENTER стоп → воспроизведение → `ENTER` сохранить / `r` перезаписать /
`e` изменить текст / `s` пропустить / `q` выйти. **VAD нет**: запись начинается и
заканчивается только по ENTER. При каждом сохранении обновляются
`manifest.jsonl`, `manifest.csv`, `dataset.json`, `README.md`. Ctrl+C завершает
сессию корректно, прогресс сохраняется — следующий запуск продолжит с первой
незаписанной позиции (`--new` начинает с нуля).

### Текущий формат streets.txt

Актуальная версия файла содержит **109 названий без слов типа улицы** (например
`Абзелиловская`, `Ак Кайын`, `Сорок лет Победы`), порядок строк = порядок записи.
Поэтому FULL reference — ровно строка файла, а «улица»/«переулок» не добавляются.
Слова типа всё равно поддерживаются парсером (если появятся в файле): они
распознаются и не дублируются. Строки-заголовки («Страница N (улицы M–K)») и
пустые строки отбрасываются и попадают в `excluded` в `dataset.json`; дубликаты и
недопустимые символы — ошибка. `streets.txt` никогда не изменяется кодом.

### Варианты записи

| variant | смысл |
|---|---|
| `FULL` | дословная строка из `streets.txt`; тип улицы (`улица`/`переулок`) **не добавляется** автоматически; reference не редактируется |
| `SHORT` | короткий вариант произношения, если он реально используется. **Вводится пользователем** после промпта `Введите короткий вариант или - чтобы пропустить:`. ENTER или `-` = пропуск записи (создаётся `status: "skipped"`, аудио не пишется). Никаких автоматических сокращений и никаких предложенных программой вариантов |
| `NATURAL` | реальная фраза целиком, **вводится пользователем** после промпта `Введите реальную фразу, которую вы будете произносить:` — с правильным падежом («До Абзелиловской», «До Хисматуллина», «На Ленина»). ENTER или `-` = пропуск. reference = введённая фраза, `reference` = её нормализация |

SHORT и NATURAL **не генерируются программой**: нет ни склонения названий, ни
подстановки предлогов. Ранее использовавшаяся автогенерация вида `До <street>`
давала грамматически неверные фразы («До Абзелиловская») и удалена вместе с
подсказками коротких форм. Перед записью reference всегда показывается на экране
(`NATURAL: До Абзелиловской`) с указанием «Произнесите ровно эту фразу.» — то,
что сохранено, всегда совпадает с тем, что реально произнесено.

Наборы, записанные до этого исправления, могли содержать автогенерированные
NATURAL-фразы (например `reference_raw: "До Абзелиловская"`). Валидация такие
фразы не помечает — она проверяет формат, а не грамматику, поэтому такой набор
считайте черновым и перезапишите его через `--new`.

### Формат и нормализация

Аудио: **mono, 8000 Hz, PCM16 WAV**. Запись идёт на native sample rate микрофона
(Mac, обычно 48000 Hz float32) → resample → 8000 Hz → PCM16. Без шумов, G.711,
сжатия, дисторшна и аугментаций — это реальные чистые записи.

Reference хранится в двух видах: `reference_raw` (как произнесено/введено) и
`reference` (для WER/CER): lowercase, пунктуация удалена, цифры → слова
(сначала проверенные числительные улиц из `forms.NUMERALS`, иначе обычные русские
числительные — например «дом 31» → «дом тридцать один»), остаются только русские
буквы и пробелы. Русские числительные в цифры не конвертируются, название улицы
не подменяется.

### Структура и метрики

```
results/real_dataset_<timestamp>/
  audio/street_001_full.wav   street_001_short.wav   street_001_natural.wav
  manifest.jsonl  manifest.csv  dataset.json  README.md
```

`manifest.jsonl`: `id`, `street_index`, `street`, `variant`, `status`,
`reference_raw`, `reference`, `audio`, `sample_rate`, `channels`, `duration_sec`.
`dataset.json` фиксирует provenance: путь и sha256 `streets.txt`, порядок улиц,
число записей, формат аудио и назначение (не fine-tuning, не hotwords).

Валидация проверяет: наличие/читаемость WAV, 8000 Hz, mono, PCM16, длительность
> 0, соответствие manifest плану (`street`/`variant`), дубликаты `id`, пустые
reference, отсутствующие/лишние файлы. Пропущенные (`status: "skipped"`) записи
ошибкой не считаются: они сохраняются в manifest как явный факт пропуска, поэтому
resume не предлагает их повторно (перезаписать пропущенную можно через
`--dataset <dir>` и удаление её строки, либо созданием нового набора).

### Эксперимент: дополнительные формы hotwords (109 реальных записей)

Отдельный экспериментальный слой; production (decoder, `hotwords.py`, веса) не меняется.

- `src/t_one_train/street_hotword_forms_experiment.json` — словарь дополнительных форм
  (`Советская → Советской`, `Искра → Искры, Искре`, `Караташ → Караташа, Караташе`, …).
  Антропонимы («Мусы Муртазина», «Файзрахмана Хисматуллина» и т. п.) и числовые названия
  («Сорок лет Победы», …) **не** склоняются: сохраняется canonical-форма.
- `src/t_one_train/street_forms_experiment.py` — чистая логика: загрузка/валидация
  словаря, сборка canonical + experimental hotwords, индекс форм улиц, детекция улиц,
  классификация изменений. Без tone/pyctcdecode — тестируется headless.
- `scripts/run_street_hotword_experiment.py` — сравнение greedy / beam без hotwords /
  beam + canonical / beam + canonical + experimental forms на 109 FULL-записях.
  Acoustic model запускается **один раз на WAV**, logprobs кэшируются в npz, и все
  конфигурации декодируются на одном и том же кэше.

```bash
uv run --group decoders python scripts/run_street_hotword_experiment.py \
    --dataset-dir results/real_dataset_2026-09-18_17-16-22 --weights 1,3,5,7,10
```

Ground-truth транскрипта у набора нет, поэтому WER/CER не считаются: метрика —
наличие произнесённого названия улицы (`street_exact_match`) и классификация изменений
относительно beam-без-hotwords (`helped` / `harmed` / `neither` / `requires_manual_review`).
Результат: `results/street_hotword_experiment_<timestamp>/`.

### Real-voice A/B/C/D тест с микрофона (Greedy / Beam / Canonical @10 / Canonical @15)

Ручной интерактивный тест: говорите фразы с улицами в микрофон и сразу видите
распознавание РОВНО четырёх конфигураций на одной записи. Акустическая модель
запускается один раз на фразу — все четыре декодера работают на одних и тех же
logprobs. Decoders и KenLM загружаются один раз на сессию.

```bash
uv run --group decoders python scripts/test_decoders.py --mic
```

Цикл: введите expected street (Enter — без оценки; используется ТОЛЬКО для
проверки результата, декодеру не передаётся) → ENTER начать запись → говорите →
ENTER остановить → блок RESULT (текст + `street: FOUND/NOT FOUND` + `changed`
/ `helped` относительно Beam) → ENTER — следующая фраза, `q` — завершить и
получить SESSION SUMMARY (correct/helped/harmed по оценённым фразам).

Каждая запись сохраняется: WAV (8 kHz mono PCM16) + per-phrase JSON +
`results.jsonl`/`results.csv` + `session_summary.json` в уникальном каталоге
`results/<YYYY-MM-DD_HH-MM-SS>/`. Тот же скрипт умеет прогонять уже записанные
файлы: `--audio путь.wav` и `--audio-dir каталог/` (те же 4 конфигурации).

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

## CPU performance benchmark (VPS)

Воспроизводимый benchmark **производительности** эталонной конфигурации T-one
на CPU (качество не меняется; используется та же конфигурация, что дала
95/109 = 87.2% reference found на реальном dataset):

- T-one (официальный ONNX, `t-tech/T-one`)
- официальный `kenlm.bin` (HF-hub)
- pyctcdecode CTC Beam Search: `beam_width=200`, `alpha=0.4`, `beta=0.9`
- Canonical hotwords из `streets.txt`, `hotword_weight=10`

### Одна команда на VPS

```bash
./scripts/run_vps_benchmark.sh
```

(эквивалент: `docker compose -f docker-compose.vps-benchmark.yml up --build --abort-on-container-exit`)

Что происходит автоматически: сборка image (`Dockerfile.vps-benchmark`) → запуск
контейнера → загрузка T-one + официального KenLM (кэшируются в Docker volume
`hf-model-cache`, повторно не скачиваются) → построение canonical hotwords из
`streets.txt` → cold start → **sequential baseline** по всем 109 FULL WAV из
`results/real_dataset_2026-09-18_17-16-22/` → sweep ORT-потоков (1/2/4/6) →
concurrency (2/4/6, 1 thread/задача) → комбинированные варианты
threads=concurrency → CPU/RAM мониторинг → `summary.json` + `summary.md` +
raw-результаты на **host** в `results/vps_benchmark_<timestamp>/`.

Параметры прогона можно переопределить без правки compose:

```bash
./scripts/run_vps_benchmark.sh --limit 5            # smoke-тест на 5 WAV
BENCHMARK_ARGS="--skip-threads" ./scripts/run_vps_benchmark.sh
```

### Метрики

Для каждого файла: `filename`, `reference`, `hypothesis`,
`audio_duration_seconds`, `model_inference_seconds` (ONNX forward),
`decoder_seconds` (beam+KenLM), `total_inference_seconds`, `RTF`,
`exact_match`, `reference_found`, `acoustic_score`/`combined_score`.
Агрегаты по каждому режиму: mean/p50/p90/p95/p99/min/max latency, RTF,
wall-clock, throughput, CPU/RAM (peak/mean). Отдельно: `startup_total_seconds`
(cold start, не входит в latency) и determinism-проверка (гипотезы всех режимов
должны совпадать с baseline).

### Файлы

| Файл | Назначение |
|---|---|
| `scripts/run_vps_benchmark.sh` | одна команда: build + run (host) |
| `docker-compose.vps-benchmark.yml` | benchmark-only compose (production не затрагивает) |
| `Dockerfile.vps-benchmark` | CPU image: tone + pyctcdecode + kenlm + onnxruntime |
| `scripts/run_vps_benchmark.py` | runner всех экспериментов |
| `src/t_one_train/vps_benchmark.py` | метрики, план экспериментов, summary/MD, CPU/RAM sampler |
| `src/t_one_train/test_decoders_core.py` | общий pipeline с quality-benchmark (`load_acoustic_model`, `load_beam_decoder`, `decode_with_scores`) |
| `tests/test_vps_benchmark.py` | headless unit-тесты метрик и плана |

Beam-width sweep (50/100/150) — опциональный CLI-режим, в автоматический прогон
не входит: `BENCHMARK_ARGS="--beam-sweep 50 100 150" ./scripts/run_vps_benchmark.sh`.

