"""Phase C, step 1 — does CAPACITY reopen the spatial-leakage gap?

LESSON — the Phase B null and why we're not done:
    Phase B trained a FROZEN ResNet18 (only a 5,130-param linear head learned)
    and found NO honest gap: random split 85.2% vs spatial split 85.6%. The
    hypothesis for that null is mechanical, not magical — the tiny head simply
    lacks the CAPACITY to memorize which places it saw in training. With so few
    parameters, the only thing it can do is learn generic class boundaries on
    top of frozen ImageNet features, which transfer to new geography just fine.

LESSON — why fine-tuning MIGHT reopen the gap:
    Here we UNFREEZE all ~11.2M parameters. Now the backbone itself can bend its
    features toward the training distribution — including its specific places.
    More capacity -> more room to memorize spatial idiosyncrasies (a particular
    river bend, a particular field texture) rather than transferable structure.
    If the Phase B hypothesis is right, the spatial holdout should now score
    LOWER than the random holdout: the gap reopens. If it stays closed even at
    full capacity, that's an even stronger "no leakage here" result. Either way
    we report the measured number honestly — same as Phase B.

LESSON — discriminative (per-group) learning rates:
    When fine-tuning a pretrained net you do NOT want one learning rate for
    everything. The backbone already holds hard-won general visual knowledge
    (edges, textures, shapes from 1.2M ImageNet photos); a large step there
    would clobber it ("catastrophic forgetting"). The fresh head, by contrast,
    starts from random noise and needs to move fast. So we give the optimizer
    TWO parameter groups: backbone at a gentle lr=1e-4 (protect what it knows)
    and the new fc head at lr=1e-3 (let it learn quickly). Same idea as a
    layer-wise LR; we just use two coarse groups for clarity.

Note: we mirror Phase B exactly — same splits (seed 42), same batch size, same
single-holdout-per-condition CPU budget — so the ONLY differences vs Phase B
are (a) the backbone is unfrozen and (b) discriminative LRs. That isolates the
effect of capacity on the leakage gap.

Run (real):  .venv/bin/python src/finetune_cv.py --epochs 3
Run (smoke): .venv/bin/python src/finetune_cv.py --epochs 1 --subset 0.02
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from train import DATA_DIR, IMAGENET_TRANSFORM, OUT_DIR, pick_device
from spatial_cv import (BATCH_SIZE, BLOCK_M, SEED, load_coords,
                        random_split_idx, spatial_split)

# Frozen-backbone results from Phase B (outputs/spatial_gap.json) — kept here
# as the comparison baseline. Verbatim in the JSON output too.
FROZEN_RANDOM_ACC = 0.8519
FROZEN_SPATIAL_ACC = 0.8561
FROZEN_GAP_ACC = -0.0043


def build_finetune_model(num_classes: int) -> nn.Module:
    """ResNet18 with DEFAULT pretrained weights, NOTHING frozen, fresh head.

    LESSON: contrast with train.build_model — there every backbone param had
    requires_grad=False. Here we leave them all trainable (the default), so the
    whole network adapts. We still swap the 1000-class ImageNet head for a
    fresh num_classes-way linear layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # No freezing loop on purpose: every parameter stays trainable.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def finetune_eval(ds, train_idx, val_idx, device, epochs,
                  lr_backbone, lr_head, tag):
    """Identical training for both splits — only the indices differ.

    Mirrors spatial_cv.train_eval but (1) builds a fully-unfrozen model and
    (2) uses Adam with two parameter groups at discriminative learning rates.
    """
    model = build_finetune_model(num_classes=len(ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    # LESSON: discriminative learning rates. The backbone (everything except
    # the new fc head) gets a GENTLE lr to protect its pretrained knowledge
    # from being overwritten; the freshly-initialized head gets a FASTER lr so
    # it can learn the EuroSAT classes quickly. Two param groups, one Adam.
    head_param_ids = {id(p) for p in model.fc.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_param_ids]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": model.fc.parameters(), "lr": lr_head},
    ])

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, persistent_workers=True)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE,
                        num_workers=2, persistent_workers=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
        # CPU full fine-tune is slow (~6-10 min/epoch expected) — print the
        # epoch wall-clock so progress is visible during long runs.
        print(f"  [{tag}] epoch {epoch}/{epochs} ({time.time() - t0:.0f}s)",
              flush=True)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return accuracy_score(targets, preds), cohen_kappa_score(targets, preds)


def subset_idx(idx, fraction):
    """Deterministically keep `fraction` of an index list (for smoke tests).

    LESSON: we subset AFTER split construction and with a fixed seed so both
    the random and spatial conditions are thinned identically — the comparison
    stays apples-to-apples even in the tiny smoke run.
    """
    if fraction >= 1.0:
        return idx
    rng = np.random.default_rng(SEED)
    keep = max(BATCH_SIZE, int(len(idx) * fraction))
    keep = min(keep, len(idx))
    chosen = rng.choice(len(idx), size=keep, replace=False)
    return [idx[i] for i in sorted(chosen)]


def plot_compare(acc_r, acc_s):
    """Grouped bars: frozen vs fine-tuned x random vs spatial accuracy."""
    conditions = ["Random split", "Spatial split"]
    frozen = [FROZEN_RANDOM_ACC, FROZEN_SPATIAL_ACC]
    finetuned = [acc_r, acc_s]

    x = np.arange(len(conditions))
    width = 0.36

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width / 2, frozen, width, label="Frozen backbone",
                color="#4fc3f7")          # light blue, dark-friendly
    b2 = ax.bar(x + width / 2, finetuned, width, label="Fine-tuned (11.2M)",
                color="#ff8a65")          # warm orange, dark-friendly

    ax.set_ylabel("Validation accuracy")
    ax.set_title("Does capacity reopen the leakage gap?\n"
                 "Frozen head vs full fine-tune, identical spatial folds")
    ax.set_xticks(x, conditions)
    ax.set_ylim(0, 1.0)
    ax.legend()
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "finetune_compare.png"
    fig.savefig(out, dpi=120)
    print(f"  saved {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--subset", type=float, default=1.0,
                        help="fraction of EACH split kept (smoke tests)")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    ds = datasets.EuroSAT(root=DATA_DIR, download=False,
                          transform=IMAGENET_TRANSFORM)
    coords = load_coords()

    # Same split policy + seed as Phase B so folds are IDENTICAL across phases.
    print("Random split:")
    tr_r, va_r = random_split_idx(len(ds))
    tr_r, va_r = subset_idx(tr_r, args.subset), subset_idx(va_r, args.subset)
    acc_r, kap_r = finetune_eval(ds, tr_r, va_r, device, args.epochs,
                                 args.lr_backbone, args.lr_head,
                                 "finetune-random")
    print(f"  random:  acc {acc_r:.4f}  kappa {kap_r:.4f}", flush=True)

    print("Spatial split:")
    tr_s, va_s = spatial_split(ds, coords)
    tr_s, va_s = subset_idx(tr_s, args.subset), subset_idx(va_s, args.subset)
    acc_s, kap_s = finetune_eval(ds, tr_s, va_s, device, args.epochs,
                                 args.lr_backbone, args.lr_head,
                                 "finetune-spatial")
    print(f"  spatial: acc {acc_s:.4f}  kappa {kap_s:.4f}", flush=True)

    plot_compare(acc_r, acc_s)

    result = {
        "mode": "finetune",
        "random": {"accuracy": round(acc_r, 4), "kappa": round(kap_r, 4)},
        "spatial": {"accuracy": round(acc_s, 4), "kappa": round(kap_s, 4)},
        "gap_accuracy": round(acc_r - acc_s, 4),
        "gap_kappa": round(kap_r - kap_s, 4),
        "epochs": args.epochs,
        "block_km": BLOCK_M // 1000,
        "frozen_reference": {
            "random_accuracy": FROZEN_RANDOM_ACC,
            "spatial_accuracy": FROZEN_SPATIAL_ACC,
            "gap_accuracy": FROZEN_GAP_ACC,
        },
        "note": ("full fine-tune, discriminative LRs (backbone 1e-4, head 1e-3); "
                 "same splits as Phase B"),
    }
    (OUT_DIR / "finetune_gap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
