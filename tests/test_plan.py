"""Plan invariants are testable without downloading or loading TTS/LLM."""
import json
import tomllib
import pytest
from src.t_one_train.tts import ROOT
from src.t_one_train.plan import build_plan, split_counts
from src.t_one_train.pipeline import records
from src.t_one_train.llm import accept_template


def test_llm_filters():
    assert accept_template('мы пока ожидаем такси на {pre}')
    assert accept_template('здравствуйте, мы пока ожидаем такси на {pre}')
    assert accept_template('добрый день, подберите нас пожалуйста на {pre}')
    for text in ['мне на Советскую', 'подберите меня на {pre}-Советскую',
                 'банан вызывает космос на {pre}', 'я стою на {pre} и на {acc}',
                 'я стою на {pre} улица ленина', 'я стою на {pre} 🐈']:
        assert not accept_template(text)


@pytest.mark.parametrize('n',[2,10,100,300,500,1000])
def test_plan_counts_and_leakage(tmp_path,n):
    cfg=tomllib.loads((ROOT/'config.toml').read_text())
    cfg['samples_per_street']=n
    llm={'accepted':['мы пока ожидаем такси на {pre}', 'я пока ожидаю такси на {pre}']}
    # Count tests use enough distinct safe LLM contexts for the 15% quota.
    llm['accepted'] += [f'{g}, {t}' for g in ['алло','здравствуйте','добрый день','добрый вечер'] for t in llm['accepted'][:]]
    out=tmp_path/'corpus'
    build_plan(cfg,ROOT,out,llm)
    counts={}
    texts=set()
    families={}
    for row in records(out/'plan.jsonl'):
        assert row['text'] not in texts
        texts.add(row['text'])
        assert row['family'] not in families or families[row['family']]==row['split']
        families[row['family']]=row['split']
        key=row['street'],row['split']
        counts[key]=counts.get(key,0)+1
    assert len(texts)==107*n
    for (street,split),count in counts.items():
        assert count==split_counts(n)[split]
    before=(out/'plan.jsonl').read_bytes()
    build_plan(cfg,ROOT,out,llm)
    assert (out/'plan.jsonl').read_bytes()==before
    cfg['seed']+=1
    with pytest.raises(ValueError,match='changed'):
        build_plan(cfg,ROOT,out,llm)
