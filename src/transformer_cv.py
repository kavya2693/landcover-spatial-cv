"""Phase C, final step — CNN vs Transformer on identical spatial folds.

LESSON — the question this benchmark answers:
    Phase B/C established the honest test: a spatial holdout (whole 50km blocks
    the model never saw) vs a random holdout. Here we ask a NEW question — does
    a fundamentally different architecture, a Vision Transformer, behave the same
    way under spatial holdout as our ResNet18 CNN? And where do they disagree at
    the per-class level? We compare on the SAME seed-42 folds so it is fair.

LESSON — what a Vision Transformer (ViT) actually does:
    ViT splits the image into 16 patches (a 4x4 grid of 16px tiles at 64px) and
    lets every patch attend to every other patch — global context from layer one.
    That is the opposite of a CNN, which slides small local filters and only
    builds up a global view slowly through depth. The patch position embeddings
    were trained at 224px (14x14 patches); timm AUTO-INTERPOLATES them down to our
    4x4 grid so the pretrained ViT runs natively at 64px with no resizing.

LESSON — per-class IoU (Jaccard), and why it is harsher than accuracy:
    For classification, per-class IoU = TP / (TP + FP + FN). Unlike accuracy it
    penalizes BOTH miss types for a class: images of the class we missed (FN) AND
    images we wrongly called the class (FP). A class can have high accuracy yet
    low IoU if it is frequently confused in either direction — so IoU exposes the
    confusable classes (e.g. AnnualCrop vs PermanentCrop) that one accuracy number
    hides.

Note: we mirror Phase B/C exactly — same splits (seed 42), same batch size, same
single-holdout-per-condition CPU budget. The ViT is fine-tuned with discriminative
learning rates (backbone 1e-4, head 1e-3) just like the CNN in finetune_cv. The
CNN here is the SAME full fine-tune as finetune_cv, re-run on the spatial split
only so we can collect its predictions for the per-class IoU comparison.

Run (real):  .venv/bin/python src/transformer_cv.py --epochs 3
Run (smoke): .venv/bin/python src/transformer_cv.py --epochs 1 --subset 0.02
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             jaccard_score)
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import DATA_DIR, IMAGENET_TRANSFORM, OUT_DIR, pick_device
from spatial_cv import (BATCH_SIZE, BLOCK_M, SEED, load_coords,
                        random_split_idx, spatial_split)
from finetune_cv import build_finetune_model, subset_idx

# Reference accuracies from Phase C (outputs/finetune_gap.json) — the full CNN
# fine-tune numbers. Kept verbatim in the JSON output for context.
CNN_REF_RANDOM_ACC = 0.9644
CNN_REF_SPATIAL_ACC = 0.9578

VIT_MODEL_NAME = "vit_tiny_patch16_224@64px (5.5M params)"
CNN_MODEL_NAME = "resnet18 full fine-tune (11.2M params)"


def build_vit_model(num_classes: int) -> nn.Module:
    """ViT-Tiny, ImageNet-pretrained, native 64px via pos-embed interpolation.

    LESSON: passing img_size=64 tells timm to interpolate the patch position
    embeddings (trained for 224px / 14x14 patches) down to our 4x4 patch grid,
    so the pretrained weights load cleanly and the model runs at native 64px —
    no upsampling of the satellite imagery required.
    """
    return timm.create_model("vit_tiny_patch16_224", pretrained=True,
                             num_classes=num_classes, img_size=64)


def _discriminative_optimizer(model, head, lr_backbone, lr_head):
    """Two Adam param groups: gentle backbone, fast fresh head.

    LESSON: discriminative learning rates (same idea as finetune_cv). The
    backbone holds hard-won pretrained features — a big step there causes
    catastrophic forgetting — so it gets a gentle lr. The freshly-attached head
    starts from random init and needs to move fast, so it gets a faster lr.
    """
    head_param_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_param_ids]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head.parameters(), "lr": lr_head},
    ])


def train_eval_preds(model, head, ds, train_idx, val_idx, device, epochs,
                     lr_backbone, lr_head, tag):
    """Fine-tune `model` and return (accuracy, kappa, preds, targets).

    Mirrors finetune_cv.finetune_eval but (1) takes any pre-built model so we
    can reuse it for ViT and CNN, and (2) RETURNS the raw predictions/targets
    so the caller can compute per-class IoU (finetune_eval returns only scalars).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _discriminative_optimizer(model, head, lr_backbone, lr_head)

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
        # CPU fine-tune is slow — print per-epoch wall-clock so long runs show
        # progress. Tag identifies which of the three runs we're in.
        print(f"  [{tag}] epoch {epoch}/{epochs} ({time.time() - t0:.0f}s)",
              flush=True)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    acc = accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds)
    return acc, kappa, preds, targets


def per_class_iou(targets, preds, classes):
    """Per-class IoU as an ordered {class_name: iou} dict, rounded 4dp.

    LESSON: jaccard_score with average=None returns one IoU per class, in label
    order (0..9). We zip those back onto the human-readable class names.
    """
    ious = jaccard_score(targets, preds, average=None,
                         labels=list(range(len(classes))))
    return {name: round(float(v), 4) for name, v in zip(classes, ious)}


def plot_compare(classes, cnn_iou, vit_iou, cnn_spatial_acc, vit_spatial_acc):
    """Grouped bars: per-class IoU, CNN vs ViT, on identical spatial folds."""
    x = np.arange(len(classes))
    width = 0.38

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width / 2, [cnn_iou[c] for c in classes], width,
                label="CNN (ResNet18)", color="#4fc3f7")   # light blue
    b2 = ax.bar(x + width / 2, [vit_iou[c] for c in classes], width,
                label="ViT (vit_tiny)", color="#ff8a65")   # warm orange

    ax.set_ylabel("Per-class IoU (Jaccard)")
    ax.set_title("CNN vs Transformer — per-class IoU, identical spatial folds")
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    # Text annotation of each model's overall spatial accuracy.
    ax.text(0.01, 0.97,
            f"Spatial accuracy:  CNN {cnn_spatial_acc:.4f}   "
            f"ViT {vit_spatial_acc:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="#222222", edgecolor="#888888"))

    fig.tight_layout()
    out = OUT_DIR / "benchmark_compare.png"
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
    print(f"Device: {device}", flush=True)
    ds = datasets.EuroSAT(root=DATA_DIR, download=False,
                          transform=IMAGENET_TRANSFORM)
    classes = ds.classes
    coords = load_coords()

    # Same split policy + seed as Phase B/C so folds are IDENTICAL across phases.
    tr_r, va_r = random_split_idx(len(ds))
    tr_r, va_r = subset_idx(tr_r, args.subset), subset_idx(va_r, args.subset)
    tr_s, va_s = spatial_split(ds, coords)
    tr_s, va_s = subset_idx(tr_s, args.subset), subset_idx(va_s, args.subset)

    # --- Run 1: ViT, random split -------------------------------------------
    print("ViT — random split:", flush=True)
    vit_r = build_vit_model(len(classes))
    vit_acc_r, vit_kap_r, _, _ = train_eval_preds(
        vit_r, vit_r.head, ds, tr_r, va_r, device, args.epochs,
        args.lr_backbone, args.lr_head, "vit-random")
    print(f"  vit random:  acc {vit_acc_r:.4f}  kappa {vit_kap_r:.4f}",
          flush=True)

    # --- Run 2: ViT, spatial split (collect preds for IoU) ------------------
    print("ViT — spatial split:", flush=True)
    vit_s = build_vit_model(len(classes))
    vit_acc_s, vit_kap_s, vit_preds, vit_targets = train_eval_preds(
        vit_s, vit_s.head, ds, tr_s, va_s, device, args.epochs,
        args.lr_backbone, args.lr_head, "vit-spatial")
    print(f"  vit spatial: acc {vit_acc_s:.4f}  kappa {vit_kap_s:.4f}",
          flush=True)
    vit_iou = per_class_iou(vit_targets, vit_preds, classes)

    # --- Run 3: CNN, spatial split (collect preds for IoU) ------------------
    # Same full fine-tune as finetune_cv (build_finetune_model + discriminative
    # LRs), re-run on the spatial fold so we can compute its per-class IoU.
    print("CNN — spatial split:", flush=True)
    cnn_s = build_finetune_model(len(classes))
    cnn_acc_s, cnn_kap_s, cnn_preds, cnn_targets = train_eval_preds(
        cnn_s, cnn_s.fc, ds, tr_s, va_s, device, args.epochs,
        args.lr_backbone, args.lr_head, "cnn-spatial")
    print(f"  cnn spatial: acc {cnn_acc_s:.4f}  kappa {cnn_kap_s:.4f}",
          flush=True)
    cnn_iou = per_class_iou(cnn_targets, cnn_preds, classes)

    plot_compare(classes, cnn_iou, vit_iou, cnn_acc_s, vit_acc_s)

    result = {
        "vit": {
            "model": VIT_MODEL_NAME,
            "random": {"accuracy": round(vit_acc_r, 4),
                       "kappa": round(vit_kap_r, 4)},
            "spatial": {"accuracy": round(vit_acc_s, 4),
                        "kappa": round(vit_kap_s, 4)},
            "gap_accuracy": round(vit_acc_r - vit_acc_s, 4),
            "spatial_per_class_iou": vit_iou,
        },
        "cnn": {
            "model": CNN_MODEL_NAME,
            "spatial": {"accuracy": round(cnn_acc_s, 4),
                        "kappa": round(cnn_kap_s, 4)},
            "spatial_per_class_iou": cnn_iou,
            "reference_from_phase_c": {
                "random_accuracy": CNN_REF_RANDOM_ACC,
                "spatial_accuracy": CNN_REF_SPATIAL_ACC,
            },
        },
        "epochs": args.epochs,
        "block_km": BLOCK_M // 1000,
        "note": ("identical seed-42 spatial folds; ViT at native 64px via "
                 "pos-embed interpolation"),
    }
    (OUT_DIR / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
