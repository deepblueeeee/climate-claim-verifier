import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from ..data import save_json
from .config import TrainingConfig
from .data import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    PairCollator,
    RobustClaimDataset,
    load_claims,
    relative_path,
)

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABELS)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )


def evaluate(model, loader, device, label_smoothing=0.0):
    model.eval()
    probs = []
    gold = []
    claim_ids = []
    loss_sum = 0.0
    loss_count = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels", None)
            ids = batch.pop("claim_ids")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            claim_ids.extend(ids)
            if labels is not None:
                labels_device = labels.to(device)
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels_device,
                    label_smoothing=label_smoothing,
                )
                loss_sum += float(loss.detach().cpu()) * labels.numel()
                loss_count += labels.numel()
                gold.extend(labels.numpy().tolist())
    probs = np.concatenate(probs, axis=0)
    pred = probs.argmax(axis=1).tolist()
    metrics = {}
    if gold:
        metrics["accuracy"] = accuracy_score(gold, pred)
        metrics["loss"] = loss_sum / max(1, loss_count)
        metrics["gold"] = gold
    metrics["pred"] = pred
    metrics["probs"] = probs
    metrics["claim_ids"] = claim_ids
    return metrics


def train_fixed_seed(args, run_dir):
    seed = args.seed
    set_seed(seed)

    train_ids, train_data = load_claims(args.train)
    dev_ids, dev_data = load_claims(args.dev)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    collator = PairCollator(tokenizer, args.max_length)
    train_ds = RobustClaimDataset(train_ids, train_data)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )

    dev_ds = RobustClaimDataset(dev_ids, dev_data)
    dev_loader = DataLoader(
        dev_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.model_name).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    def lr_factor(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return max(
            0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    checkpoint_dir = run_dir / f"seed_{seed}_epoch_{args.epochs}"
    epoch_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        start = time.time()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch.pop("claim_ids")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                )
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_loss = float(np.mean(losses))
        elapsed = time.time() - start
        epoch_records.append(
            {"epoch": epoch, "train_loss": train_loss, "elapsed_seconds": elapsed}
        )

        print(
            f"{seed} | {epoch} | {train_loss:.4f} | {elapsed:.1f}s",
            flush=True,
        )

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    metrics = evaluate(
        model,
        dev_loader,
        device,
        label_smoothing=args.label_smoothing,
    )

    print(f"\nDev accuracy: {metrics['accuracy']:.4f}")
    print(
        classification_report(
            metrics["gold"],
            metrics["pred"],
            labels=range(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    return {
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "dev_loss": metrics["loss"],
        "epoch": args.epochs,
        "train_loss": epoch_records[-1]["train_loss"] if epoch_records else None,
        "checkpoint": relative_path(checkpoint_dir),
        "epoch_records": epoch_records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Robust RoBERTa classifier for retrieved claim evidence."
    )
    parser.add_argument(
        "--train", default="data/classifier_inputs/reranker_top5/train.json"
    )
    parser.add_argument(
        "--dev", default="data/classifier_inputs/reranker_top5/dev.json"
    )
    parser.add_argument("--run-dir", default="predictions/robust_roberta_runs")
    parser.add_argument("--model-name", default=TrainingConfig.model_name)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainingConfig.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--label-smoothing", type=float, default=TrainingConfig.label_smoothing)
    return train(TrainingConfig(**vars(parser.parse_args(argv))))


def train(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_summary = train_fixed_seed(args, run_dir)
    summary_path = run_dir / "summary.json"
    saved_args = vars(args).copy()
    for path_key in ("train", "dev", "run_dir"):
        saved_args[path_key] = relative_path(saved_args[path_key])
    save_json(summary_path, {"runs": [run_summary], "args": saved_args})

    print(f"\nFinal checkpoint: {run_summary['checkpoint']}")


if __name__ == "__main__":
    main()
