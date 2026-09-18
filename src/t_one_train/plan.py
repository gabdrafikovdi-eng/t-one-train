"""Immutable disk plan: distinct normalized texts and globally disjoint families."""
import hashlib
import json
import random
from .forms import read_streets, render, normalize
from .templates import candidates, PREFIXES, TAILS
from .tts import VOICES


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_counts(n):
    if n < 2:
        raise ValueError("samples-per-street must be >=2")
    if n == 2:
        return {"train": 1, "validation": 0, "test": 1}
    holdout = max(1, round(n*.05))
    return {"train": n-2*holdout, "validation": holdout, "test": holdout}


def build_plan(cfg, root, out, llm):
    streets, excluded = read_streets(root / "streets.txt")
    if len(streets) != cfg["expected_streets"]:
        raise ValueError(f"Expected {cfg['expected_streets']} streets, got {len(streets)}")
    spec = dict(config=cfg, streets=streets, excluded=excluded, llm=llm["accepted"], version=1)
    # Freeze source hashes too: resume cannot silently mix changed algorithms.
    spec["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root/'src'/'t_one_train').glob('*.py'))}
    fingerprint = digest(spec)
    spec["fingerprint"] = fingerprint
    spec_path = out / "run.json"
    if spec_path.exists():
        previous = json.loads(spec_path.read_text())
        if previous != spec:
            raise ValueError("Run config, code, streets or LLM changed: use another output directory")
        plan_path = out / "plan.jsonl"
        expected = json.loads((out / "plan.sha256.json").read_text())
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Plan integrity check failed")
        return spec
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.jsonl.tmp"
    used = set()
    profiles = list(cfg['profiles'])
    weights = [cfg['profiles'][p]['weight'] for p in profiles]
    with path.open('w') as f:
        for si, street in enumerate(streets):
            for split, count in split_counts(cfg['samples_per_street']).items():
                rng = random.Random(f"{cfg['seed']}:{street}:{split}")
                bank = [(t, family, "templates") for t, family in candidates(split)]
                if split == 'train':
                    for i, t in enumerate(llm['accepted']):
                        for tail in TAILS[:8]:
                            bank.append((t + (', '+tail if tail and tail not in t else ''), f'llm:{i}', 'llm'))
                rng.shuffle(bank)
                if split == 'train' and count >= 100:
                    # Guarantee genuine short utterances instead of letting a large
                    # combinatorial context bank almost eliminate them by chance.
                    short = [r for r in bank if len(r[0].split()) <= 2 and r[2]=='templates']
                    bank = short + [r for r in bank if r not in short]
                # Reserve at least 15% of training rows for actual local LLM contexts.
                if split == 'train':
                    target = max(1, int(count*.10))
                    bank = [r for r in bank if r[2]=='llm'][:target*3] + [r for r in bank if r[2]!='llm'] + [r for r in bank if r[2]=='llm']
                selected = 0
                llm_count = 0
                for template, family, source in bank:
                    if selected == count:
                        break
                    if source == 'llm' and llm_count >= max(1,int(count*.10)):
                        continue
                    spoken = render(template, street)
                    if not spoken or len(spoken.split()) > 23:
                        continue
                    text = normalize(spoken)
                    if text in used:
                        continue
                    used.add(text)
                    ident = digest([fingerprint, street, split, text])[:24]
                    row = dict(id=ident, audio=f'audio/{split}/{si:03d}/{ident}.wav', text=text,
                               tts_text=spoken+'.', street=street, split=split, family=family,
                               template=template, source=source, voice=VOICES[(si+selected)%len(VOICES)],
                               seed=int(ident[:8],16), speed=round(rng.uniform(cfg['augmentation']['speed_min'],cfg['augmentation']['speed_max']),3),
                               profile=rng.choices(profiles, weights)[0])
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    selected += 1
                    llm_count += source == 'llm'
                if selected != count or (split == 'train' and count and llm_count == 0):
                    raise ValueError(f'Insufficient distinct texts: {street} {split}: {selected}/{count}; llm={llm_count}')
    path.replace(out/'plan.jsonl')
    (out/'plan.sha256.json').write_text(json.dumps(hashlib.sha256((out/'plan.jsonl').read_bytes()).hexdigest()))
    spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    return spec
