#!/usr/bin/env python3
"""
Baseline expression classifier on depth clips.

    python train_baseline.py                       # train on dataset/
    python train_baseline.py --mode deform --epochs 40

A small 3D CNN over (2, T, S, S) inputs -- depth + validity mask. Deliberately
small (~0.4 M params) because early datasets are small; it exists to prove the
data pipeline end to end and to give you a real number to beat, not to be the
final model. The checkpoint stores the label map and the full preprocessing
contract, so live_infer.py can reproduce training conditions exactly.

Scaling up later: swap the model for an R(2+1)D / video transformer, or
pretrain on unlabeled clips (build_dataset.py --include-unlabeled) and
fine-tune. Nothing else in the pipeline needs to change.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("training needs PyTorch:\n"
                     "  pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu")

from depth_dataset import DepthClipDataset, make_loader
from preprocessing import PREPROC_VERSION

HERE = Path(__file__).resolve().parent


class DepthExpressionNet(nn.Module):
    """3D CNN: spatial-temporal features over a short depth clip."""

    def __init__(self, n_classes, in_ch=2, width=16, dropout=0.3):
        super().__init__()

        def block(cin, cout, pool):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.MaxPool3d(pool))

        self.features = nn.Sequential(
            block(in_ch, width, (1, 2, 2)),        # keep time, halve space
            block(width, width * 2, (2, 2, 2)),
            block(width * 2, width * 4, (2, 2, 2)),
            block(width * 4, width * 8, (2, 2, 2)),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(width * 8, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, loader, device, n_classes, weights=None):
    """Uses the SAME criterion as training (weighted if training is weighted),
    so the reported train and val losses are on one scale and comparable."""
    model.eval()
    conf = np.zeros((n_classes, n_classes), np.int64)
    loss_sum, n = 0.0, 0
    lossf = nn.CrossEntropyLoss(weight=weights)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(lossf(logits, y)) * y.numel()
            n += y.numel()
            for t, p in zip(y.cpu().numpy(), logits.argmax(1).cpu().numpy()):
                conf[t, p] += 1
    acc = float(np.trace(conf) / max(1, conf.sum()))
    # macro F1 over classes actually present in this split
    f1s = []
    for c in range(n_classes):
        if conf[c].sum() == 0:
            continue
        tp, fp = conf[c, c], conf[:, c].sum() - conf[c, c]
        fn = conf[c].sum() - conf[c, c]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"loss": loss_sum / max(1, n), "acc": acc,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0, "conf": conf}


def print_confusion(conf, labels):
    present = [i for i in range(len(labels)) if conf[i].sum() or conf[:, i].sum()]
    if not present:
        return
    w = max(len(labels[i]) for i in present) + 1
    print("    " + "true\\pred".ljust(w) + "".join(
        f"{labels[i][:6]:>7s}" for i in present))
    for i in present:
        print("    " + labels[i].ljust(w) + "".join(
            f"{conf[i, j]:7d}" for j in present))


def main():
    p = argparse.ArgumentParser(description="Train the depth expression baseline")
    p.add_argument("--dataset", default=str(HERE / "dataset"))
    p.add_argument("--out", default=str(HERE / "models"))
    p.add_argument("--mode", choices=["absolute", "deform"], default="deform",
                   help="'deform' subtracts the session neutral face "
                        "(usually stronger and more subject-invariant)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--drop-unlabeled", action="store_true", default=True,
                   help="exclude the 'unlabeled' class from training")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.dataset)
    spec = json.loads((root / "dataset_spec.json").read_text(encoding="utf-8"))
    all_labels = [k for k, _ in sorted(spec["label_map"].items(),
                                       key=lambda kv: kv[1])]
    labels = [l for l in all_labels if l != "unlabeled"] \
        if args.drop_unlabeled else all_labels
    if len(labels) < 2:
        raise SystemExit(
            f"need >= 2 labelled classes to train, dataset has {labels}.\n"
            f"Record sessions with the 1-9 label keys pressed, then re-run "
            f"build_dataset.py")

    common = dict(mode=args.mode, labels=labels)
    train_ds = DepthClipDataset(root, split="train",
                               augment=not args.no_augment, **common)
    val_ds = DepthClipDataset(root, split="val", augment=False, **common)
    test_ds = DepthClipDataset(root, split="test", augment=False, **common)
    if not len(train_ds):
        raise SystemExit("train split has no labelled clips")

    print(f"labels: {labels}")
    print(f"clips  train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)}   mode={args.mode}")
    print(f"train class counts: "
          f"{dict(zip(labels, train_ds.class_counts()))}")
    if not len(val_ds):
        print("! no val clips -- selecting the final epoch instead of the "
              "best, and any accuracy below is training-set only")

    device = torch.device(args.device)
    model = DepthExpressionNet(len(labels), width=args.width).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par/1e3:.0f}k params on {device}")

    weights = torch.tensor(train_ds.class_weights(), device=device)
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    train_loader = make_loader(train_ds, args.batch_size, True,
                              args.num_workers)
    val_loader = (make_loader(val_ds, args.batch_size, False, args.num_workers)
                  if len(val_ds) else None)
    test_loader = (make_loader(test_ds, args.batch_size, False,
                               args.num_workers) if len(test_ds) else None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "expression_model.pt"
    best = {"macro_f1": -1.0, "epoch": -1}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = lossf(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * y.numel()
            seen += y.numel()
            correct += int((logits.argmax(1) == y).sum())
        sched.step()
        tr = {"loss": tot / max(1, seen), "acc": correct / max(1, seen)}
        line = (f"epoch {epoch:3d}/{args.epochs}  "
                f"train loss {tr['loss']:.3f} acc {tr['acc']:.3f}")

        va = evaluate(model, val_loader, device, len(labels), weights) \
            if val_loader else None
        if va:
            line += (f"   val loss {va['loss']:.3f} acc {va['acc']:.3f} "
                     f"macroF1 {va['macro_f1']:.3f}")
        print(line)
        history.append({"epoch": epoch, "train": tr,
                        "val": {k: v for k, v in (va or {}).items()
                                if k != "conf"}})

        score = va["macro_f1"] if va else tr["acc"]
        is_best = score > best["macro_f1"]
        if is_best or (va is None and epoch == args.epochs):
            best = {"macro_f1": score, "epoch": epoch}
            torch.save({
                "model_state": model.state_dict(),
                "arch": {"name": "DepthExpressionNet", "width": args.width,
                         "in_ch": 2, "n_classes": len(labels)},
                "labels": labels,
                "mode": args.mode,
                "clip_len": int(spec["clip"]["length"]),
                "preproc_version": PREPROC_VERSION,
                "preproc_params": spec["preproc_params"],
                "dataset_spec": {k: spec[k] for k in
                                 ("preproc_params", "clip", "counts",
                                  "split_sessions")},
                "selected_on": "val_macro_f1" if va else "final_epoch",
                "epoch": epoch,
            }, ckpt_path)

    print(f"\nbest epoch {best['epoch']} "
          f"({'val macroF1' if val_loader else 'train acc'} "
          f"{best['macro_f1']:.3f}) -> {ckpt_path}")

    if test_loader:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        te = evaluate(model, test_loader, device, len(labels), weights)
        print(f"\nTEST  acc {te['acc']:.3f}  macroF1 {te['macro_f1']:.3f}  "
              f"({len(test_ds)} clips from sessions "
              f"{spec['split_sessions']['test']})")
        print_confusion(te["conf"], labels)
        history.append({"test": {k: v for k, v in te.items() if k != "conf"},
                        "test_confusion": te["conf"].tolist()})
    else:
        print("\nno test split -- record more sessions for an honest number "
              "(splits are grouped by session)")

    (out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nrun live:  python live_infer.py --model {ckpt_path}")


if __name__ == "__main__":
    main()
