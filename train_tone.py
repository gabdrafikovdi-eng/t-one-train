"""Official T-one fine-tuning / evaluation. Run on the CUDA PC, NOT on the Mac."""
import argparse
import json
import os
import re
import numpy as np
import torch
import evaluate
from datasets import Audio, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from tone.training.data_collator import DataCollatorCTCWithPadding
from tone.training.model_wrapper import ToneForCTC

SAMPLE_RATE = 8000
REG = re.compile("[а-яё]+")
PAD = round(SAMPLE_RATE * 0.3)  # official notebook: 300 ms of zeros on both sides


def load_manifests(paths):
    dataset = load_dataset("json", data_files=paths)
    for split in dataset:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return dataset


def prepare_dataset(processor, batch):
    audio = batch["audio"]
    values = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_values"] = np.pad(values, (PAD, PAD), mode="constant")
    batch["input_lengths"] = len(batch["input_values"])
    text = " ".join(REG.findall(batch["text"].lower()))
    batch["labels"] = processor(text=text).input_ids
    return batch


def build_dataset(processor, paths, num_proc):
    dataset = load_manifests(paths)
    dataset = dataset.map(prepare_dataset, fn_kwargs={"processor": processor},
                          remove_columns=["audio", "text"], num_proc=num_proc)
    return dataset.filter(lambda x: x < 20.0 * SAMPLE_RATE, input_columns=["input_lengths"])


def compute_metrics(processor, wer_metric, preds):
    pred_ids = np.argmax(preds.predictions, axis=-1)
    pred_ids[(preds.predictions == -100).all(axis=-1)] = processor.tokenizer.pad_token_id
    preds.label_ids[preds.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(preds.label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train")
    parser.add_argument("--validation")
    parser.add_argument("--eval-only")
    parser.add_argument("--save-predictions", help="write predictions JSONL aligned with --eval-only order")
    parser.add_argument("--model", default="t-tech/T-one", help="checkpoint for --eval-only")
    parser.add_argument("--output", default="tone_ctc")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = Wav2Vec2Processor(
        feature_extractor=Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE,
            padding_value=0.0, return_attention_mask=False, do_normalize=False),
        tokenizer=Wav2Vec2CTCTokenizer.from_pretrained("t-tech/T-one", pad_token="[PAD]", word_delimiter_token="|"))
    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = evaluate.load("wer")
    dataset = build_dataset(processor, {s: p for s, p in
        (("train", args.train), ("validation", args.validation), ("test", args.eval_only)) if p}, os.cpu_count())
    if args.eval_only:
        model = ToneForCTC.from_pretrained(args.model)
        trainer = Trainer(model=model, data_collator=collator,
            compute_metrics=lambda p: compute_metrics(processor, wer_metric, p),
            eval_dataset=dataset["test"])
        print(trainer.evaluate())
        if args.save_predictions:
            prediction = trainer.predict(dataset["test"])
            ids = np.argmax(prediction.predictions, axis=-1)
            ids[(prediction.predictions == -100).all(axis=-1)] = processor.tokenizer.pad_token_id
            texts = processor.batch_decode(ids)
            references = [json.loads(line) for line in open(args.eval_only, encoding="utf-8")]
            with open(args.save_predictions, "w", encoding="utf-8") as f:
                for i, (text, row) in enumerate(zip(texts, references)):
                    f.write(json.dumps({"index": i, "audio": row["audio"], "text": row["text"],
                        "street": row.get("street"), "prediction": text}, ensure_ascii=False) + "\n")
        return
    model = ToneForCTC.from_pretrained("t-tech/T-one")
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=4,
        eval_on_start=True,
        num_train_epochs=args.epochs,
        bf16=True,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=1e-6,
        warmup_ratio=0.05,
        save_total_limit=2,
    )
    trainer = Trainer(model=model, data_collator=collator,
        compute_metrics=lambda p: compute_metrics(processor, wer_metric, p),
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor)
    trainer.train()


if __name__ == "__main__":
    main()
