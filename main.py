"""Synthetic corpus CLI. Never starts T-one training."""
import argparse
import json
from pathlib import Path
import tomllib

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','validate','stats','prepare-texts'])
    parser.add_argument('--samples-per-street',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'config.toml')
    args=parser.parse_args()
    cfg=tomllib.loads(args.config.read_text())
    if args.samples_per_street is not None:
        cfg['samples_per_street']=args.samples_per_street
    out=args.output or ROOT/(cfg['output'] if cfg['samples_per_street']==500 else f"dataset_{cfg['samples_per_street']}")
    out=out.resolve()
    if args.command=='prepare-texts':
        from src.t_one_train.llm import prepare
        data=prepare(cfg,ROOT)
        print(f"Accepted LLM templates: {len(data['accepted'])}")
    elif args.command=='generate':
        from src.t_one_train.pipeline import generate
        from src.t_one_train.quality import validate
        generate(cfg,ROOT,out)
        validate(out)
    elif args.command=='validate':
        from src.t_one_train.quality import validate
        validate(out)
    else:
        from src.t_one_train.quality import show
        show(json.loads((out/'report.json').read_text()))


if __name__=='__main__':
    main()

