"""Streaming generation and atomic sidecars for crash-safe resume."""
import contextlib
import fcntl
import json
import os
import time
from pathlib import Path
import soundfile as sf
from .audio import telephone, inspect_audio
from .tts import Silero
from .llm import prepare
from .plan import build_plan


def records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False), encoding='utf-8')
    os.replace(tmp,path)


@contextlib.contextmanager
def lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another generator is writing {out}') from None
        yield


def cached(row, out):
    path = out/row['audio']
    side = path.with_suffix('.json')
    if not path.exists() or not side.exists():
        return None
    try:
        data = json.loads(side.read_text())
        if any(data.get(k)!=v for k,v in row.items()):
            return None
        info = inspect_audio(path)
        if any(data.get(k)!=v for k,v in info.items()):
            return None
        return data
    except (ValueError, RuntimeError, OSError):
        return None


def export(out):
    with contextlib.ExitStack() as stack:
        files = {s:stack.enter_context((out/f'{s}.jsonl.tmp').open('w')) for s in ['train','validation','test']}
        meta = stack.enter_context((out/'metadata.jsonl.tmp').open('w'))
        for row in records(out/'plan.jsonl'):
            data = cached(row,out)
            if data is None:
                raise ValueError(f'Missing/corrupt sample {row["id"]}')
            files[row['split']].write(json.dumps({k:row[k] for k in ['audio','text']},ensure_ascii=False)+'\n')
            meta.write(json.dumps(data,ensure_ascii=False)+'\n')
    for name in ['train','validation','test','metadata']:
        os.replace(out/f'{name}.jsonl.tmp',out/f'{name}.jsonl')


def generate(cfg,root,out):
    with lock(out):
        llm = prepare(cfg,root)
        spec = build_plan(cfg,root,out,llm)
        model = None
        start=time.monotonic()
        total=len(spec['streets'])*cfg['samples_per_street']
        reused=0
        errors=0
        for n,row in enumerate(records(out/'plan.jsonl'),1):
            if cached(row,out):
                reused+=1
                continue
            if model is None:
                model=Silero(cfg['device'],cfg['threads'])
                provenance=model.provenance()
                provenance['run_fingerprint']=spec['fingerprint']
                old=out/'tts.json'
                if old.exists() and json.loads(old.read_text())['sha256']!=provenance['sha256']:
                    raise ValueError('Silero model hash changed; cannot resume')
                atomic_json(old,provenance)
            path=out/row['audio']
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                raw=model.speak(row['tts_text'],row['voice'],row['seed'])
                audio,aug=telephone(raw,row['speed'],row['profile'],cfg,row['seed'])
                tmp=path.with_suffix('.wav.tmp')
                sf.write(tmp,audio,8000,subtype='PCM_16',format='WAV')
                inspect_audio(tmp)
                os.replace(tmp,path)
                info=inspect_audio(path)
                atomic_json(path.with_suffix('.json'),dict(**row,**aug,**info,device=model.device))
            except Exception as e:
                errors+=1
                with (out/'errors.jsonl').open('a') as f:
                    f.write(json.dumps(dict(id=row['id'],error=repr(e),time=time.time()))+'\n')
                # Fail closed rather than silently deliver fewer samples.
                raise
            if n%100==0 or n==total:
                elapsed=time.monotonic()-start
                rate=(n-reused)/max(elapsed,.001)
                print(f'{n}/{total} reused={reused} errors={errors} samples/s={rate:.2f} ETA={(total-n)/max(rate,.001)/60:.1f}min',flush=True)
        export(out)
        atomic_json(out/'complete.json',dict(total=total,reused=reused,elapsed=time.monotonic()-start,
                                            fingerprint=spec['fingerprint']))
        print(f'Generation complete: {total} samples; reused={reused}',flush=True)
