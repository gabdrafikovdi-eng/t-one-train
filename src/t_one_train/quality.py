"""Read every WAV; check counts, manifests, source/family and content leakage."""
from collections import Counter
import hashlib
import json
import statistics
from .pipeline import records, atomic_json
from .audio import inspect_audio
from .forms import normalize, render
from .plan import split_counts


def validate(out):
    spec=json.loads((out/'run.json').read_text())
    expected=len(spec['streets'])*spec['config']['samples_per_street']
    if hashlib.sha256((out/'plan.jsonl').read_bytes()).hexdigest()!=json.loads((out/'plan.sha256.json').read_text()):
        raise ValueError('Plan hash mismatch')
    counts={k:Counter() for k in ['street','voice','profile','split','source','codec']}
    texts, hashes, families, ids = {}, {}, {}, set()
    durations=[]
    invalid=[]
    leakage=[]
    audio_duplicates=0
    text_duplicates=0
    total_bytes=0
    per_street_split=Counter()
    manifests={s:iter(records(out/f'{s}.jsonl')) for s in ['train','validation','test']}
    metadata=iter(records(out/'metadata.jsonl'))
    for row in records(out/'plan.jsonl'):
        try:
            m=next(metadata)
            if any(m.get(k)!=v for k,v in row.items()):
                raise ValueError('Metadata disagrees with plan')
            manifest=next(manifests[row['split']])
            if manifest!={k:row[k] for k in ['audio','text']}:
                raise ValueError('Manifest disagrees with plan')
            if normalize(render(row['template'],row['street']))!=row['text']:
                raise ValueError('Street/template mismatch')
            info=inspect_audio(out/row['audio'])
            if any(m.get(k)!=v for k,v in info.items()):
                raise ValueError('Audio changed after generation')
            if row['id'] in ids:
                raise ValueError('Duplicate ID')
            ids.add(row['id'])
            durations.append(info['duration'])
            total_bytes+=info['bytes']
            for key in counts:
                counts[key][m[key]]+=1
            per_street_split[row['street'],row['split']]+=1
            for key, value, seen in [('text',row['text'],texts),('audio',info['audio_sha256'],hashes),('family',row['family'],families)]:
                if value in seen and seen[value]!=row['split']:
                    leakage.append(dict(id=row['id'],kind=key))
                if key=='text' and value in seen:
                    text_duplicates+=1
                if key=='audio' and value in seen:
                    audio_duplicates+=1
                seen[value]=row['split']
        except Exception as e:
            invalid.append(dict(id=row['id'],error=str(e)))
    for name,it in list(manifests.items())+[('metadata',metadata)]:
        if next(it,None) is not None:
            invalid.append(dict(error=f'Extra rows in {name}'))
    for street in spec['streets']:
        for split,count in split_counts(spec['config']['samples_per_street']).items():
            if per_street_split[street,split]!=count:
                invalid.append(dict(error=f'Count mismatch: {street} {split}'))
    disk_wavs=sum(1 for _ in (out/'audio').rglob('*.wav'))
    if disk_wavs!=expected:
        invalid.append(dict(error=f'WAV count {disk_wavs} != {expected}'))
    report=dict(streets=len(counts['street']),samples=len(durations),expected=expected,
        sample_rate=8000,channels=1,counts={k:dict(v) for k,v in counts.items()},
        invalid=invalid,leakage=leakage,duplicate_text=text_duplicates,duplicate_audio=audio_duplicates,
        duration=dict(min=min(durations,default=0),max=max(durations,default=0),
                      mean=statistics.mean(durations) if durations else 0,
                      median=statistics.median(durations) if durations else 0,total=sum(durations)),
        wav_bytes=total_bytes,excluded=spec['excluded'],
        generation_errors=sum(1 for _ in records(out/'errors.jsonl')) if (out/'errors.jsonl').exists() else 0)
    report['passed']=not invalid and not leakage and not audio_duplicates and not text_duplicates and len(durations)==expected
    atomic_json(out/'report.json',report)
    show(report)
    if not report['passed']:
        raise ValueError(f'Dataset validation failed; see {out / "report.json"}')
    return report


def show(report):
    summary={k:v for k,v in report.items() if k!='counts'}
    summary['counts']={k:v for k,v in report['counts'].items() if k!='street'}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Per-street counts are in report.json: counts.street',flush=True)
