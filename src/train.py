"""Step 2 — Train a baseline land-cover classifier (transfer learning).

LESSON — the whole of supervised learning in one loop:
    1. GUESS:   the model predicts a class for a batch of images
    2. MEASURE: the loss function scores how wrong the guesses were
    3. ADJUST:  the optimizer nudges the model's weights to be less wrong
    Repeat over the dataset (one full pass = an "epoch") until good.

LESSON — transfer learning (the trick that makes today possible):
    ResNet18 was already trained on 1.2M ImageNet photos and learned general
    visual features (edges, textures, shapes). We FREEZE all of that and
    replace only its final layer with a fresh one that outputs 10 EuroSAT
    classes. Only that tiny layer learns. Result: minutes on a laptop CPU
    instead of days on a GPU, with far less data needed.

Run:  .venv/bin/python src/train.py --epochs 3
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             cohen_kappa_score, confusion_matrix)
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def pick_device() -> torch.device:
    """Prefer Apple GPU (MPS) when available, otherwise CPU.

    LESSON: PyTorch code is device-agnostic — the same code runs on
    CUDA/MPS/CPU; you just move model and data to the chosen device.
    (This Mac runs macOS 13, and PyTorch 2.5+ needs macOS 14 for MPS,
    so today it falls back to CPU. The frozen backbone keeps that fast.)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze: pretrained features stay fixed
    # Replace the 1000-class ImageNet head with a fresh 10-class one.
    # New layers default to requires_grad=True, so only this layer trains.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=float, default=1.0,
                        help="fraction of data to use (e.g. 0.2 for a quick run)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # LESSON: normalization. The pretrained ResNet expects inputs scaled the
    # same way ImageNet was — so we reuse ImageNet's channel means/stds.
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.EuroSAT(root=DATA_DIR, download=True, transform=tf)
    classes = ds.classes  # grab before random_split wraps ds in a Subset

    if args.subset < 1.0:
        keep = int(len(ds) * args.subset)
        ds, _ = random_split(ds, [keep, len(ds) - keep],
                             generator=torch.Generator().manual_seed(42))

    # LESSON: train/validation split. We hold out 20% the model NEVER trains
    # on, to measure how well it generalizes to unseen images. (Phase B will
    # show why even this split can be too optimistic for satellite data —
    # nearby patches leak information across the split: spatial autocorrelation.)
    n_val = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # num_workers: decode/normalize images in parallel processes so the CPU
    # training loop isn't stuck waiting on JPEG decoding between batches.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=2, persistent_workers=True)

    model = build_model(num_classes=len(classes)).to(device)

    # LESSON: cross-entropy loss = "how surprised was the model by the true
    # label" — the standard loss for classification. Adam = an optimizer that
    # adapts each weight's step size; lr controls overall step size.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running_loss = time.time(), 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()          # clear old gradients
            loss = criterion(model(images), labels)  # GUESS + MEASURE
            loss.backward()                # compute how to adjust
            optimizer.step()               # ADJUST
            running_loss += loss.item() * len(images)

        # Evaluate on held-out data (no learning here — eval mode, no grads)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in val_dl:
                out = model(images.to(device))
                preds.extend(out.argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())

        acc = accuracy_score(targets, preds)
        kappa = cohen_kappa_score(targets, preds)
        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss {running_loss / len(train_ds):.4f} | "
              f"val acc {acc:.4f} | kappa {kappa:.4f} | "
              f"{time.time() - t0:.0f}s")

    # LESSON: a confusion matrix shows WHICH classes get mixed up (e.g.
    # PermanentCrop vs AnnualCrop) — far more informative than one number.
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"EuroSAT baseline — val acc {acc:.3f}, kappa {kappa:.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)

    metrics = {"val_accuracy": round(acc, 4), "cohen_kappa": round(kappa, 4),
               "epochs": args.epochs, "subset": args.subset,
               "model": "resnet18-frozen-backbone"}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {OUT_DIR / 'confusion_matrix.png'} and metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
