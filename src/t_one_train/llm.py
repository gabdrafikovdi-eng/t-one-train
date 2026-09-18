"""Local LLM contexts, conservatively filtered; never ask LLM for street names."""
import json
import re
import time
import urllib.request
from pathlib import Path
from .forms import normalize
from .templates import BANKS

# Auditable semantic grammar: only taxi/location clauses, one typed address slot.
# Unknown syntax is rejected rather than claiming to detect arbitrary nonsense.
SAFE = re.compile(
    r"(?:(?:алло|здравствуйте|добрый день|добрый вечер)[, ]+)?"
    r"(?:"
    r"(?:я (?:сейчас |пока |уже )?(?:ожидаю|стою|жду|буду ждать)|мы (?:сейчас |пока |уже )?(?:ожидаем|стоим|ждём|будем ждать)) (?:такси )?на \{pre\}"
    r"|(?:пришлите|отправьте|вызовите|закажите) (?:мне |нам )?(?:пожалуйста )?(?:машину|такси) на \{acc\}"
    r"|(?:мне|нам) (?:пожалуйста )?(?:нужна машина|нужно такси|нужно доехать) до \{gen\}"
    r"|(?:хочу|хотим) (?:заказать|вызвать) (?:машину|такси) на \{acc\}"
    r"|(?:заберите|подберите) (?:меня|нас) (?:пожалуйста )?на \{pre\}"
    r")"
    r"(?:[, ]+(?:пожалуйста|я буду ждать|я уже на месте|позвоните когда подъедете))?[.!?]?"
)


def accept_template(text):
    if not isinstance(text, str) or not 3 <= len(text.split()) <= 18 or len(text) > 180:
        return False
    if not SAFE.fullmatch(text.lower().strip()):
        return False
    skeleton = normalize(re.sub(r"\{\w+\}", "адрес", text))
    known = {normalize(re.sub(r"\{\w+\}", "адрес", t)) for bank in BANKS.values() for t in bank}
    return skeleton not in known


def request(url, payload, attempts=3):
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after {attempts} attempts: {last}")


def prepare(cfg, root):
    c = cfg["llm"]
    path = root / c["cache"]
    if path.exists():
        data = json.loads(path.read_text())
        if not data["accepted"] or not all(accept_template(t) for t in data["accepted"]):
            raise ValueError("Invalid cached LLM templates")
        return data
    accepted, responses, rejected = set(), [], []
    seeds = [
        'мы пока ожидаем такси на {pre}',
        'подберите нас пожалуйста на {pre}',
        'пришлите нам пожалуйста машину на {acc}',
        'нам нужно такси до {gen}',
        'я сейчас стою на {pre}',
        'хотим вызвать такси на {acc}',
        'отправьте нам такси на {acc}',
        'мы уже ждём такси на {pre}',
    ]
    for i, seed_text in enumerate(seeds):
        prompt = ('Return JSON {"templates": [strings]}. Write 12 different natural Russian taxi requests. '
            'Each item must be a complete sentence with exactly one literal placeholder. '
            'Use {pre} after на for pickup location; {acc} after на for destination; {gen} after до. '
            f'Base sentence: "{seed_text}". '
            'Examples: "здравствуйте, мы пока ожидаем такси на {pre}", '
            '"добрый день, пришлите нам пожалуйста машину на {acc}", '
            '"алло, нам нужно такси до {gen}". '
            'Vary greeting (алло, здравствуйте, добрый день, добрый вечер), '
            'or append я буду ждать / я уже на месте / позвоните когда подъедете. '
            'No real addresses, names, numbers, phone numbers. No extra placeholders.')
        response = request(c["url"] + "/api/generate", dict(model=c["model"], prompt=prompt,
            stream=False, format={"type":"object","properties":{"templates":{"type":"array","items":{"type":"string"}}},"required":["templates"]}, keep_alive="0", options=dict(seed=cfg["seed"]+i,
            temperature=.7, num_ctx=2048, num_predict=2200)))
        responses.append(response)
        try:
            values = json.loads(response["response"])["templates"]
        except (ValueError, KeyError, TypeError):
            continue
        for t in values:
            if accept_template(t):
                accepted.add(t.lower().strip().rstrip(".!?"))
            else:
                rejected.append(t)
        if len(accepted) >= 20:
            break
    if len(accepted) < 8:
        path.parent.mkdir(exist_ok=True)
        (path.parent/'llm_rejected_attempt.json').write_text(json.dumps(dict(accepted=sorted(accepted), rejected=rejected, responses=responses),ensure_ascii=False,indent=2))
        raise RuntimeError(f"Only {len(accepted)} safe LLM contexts; see assets/llm_rejected_attempt.json")
    tags = request(c["url"]+"/api/show", {"model": c["model"]})
    data = dict(model=c["model"], accepted=sorted(accepted), rejected=rejected,
                responses=responses, model_info=tags, seed=cfg["seed"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data
