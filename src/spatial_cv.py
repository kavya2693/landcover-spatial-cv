"""Phase B, step 2 — the honest test: random split vs spatial split.

LESSON: a random split scatters validation patches BETWEEN training patches.
Since nearby satellite patches are nearly identical (spatial autocorrelation),
the model can score well by recognizing PLACES it trained on. A spatial split
holds out whole 50km blocks the model has never seen any part of — measuring
true generalization to NEW geography. The accuracy drop between the two is
the "honest gap" this project exists to report.

Note: we run ONE spatial holdout vs ONE random holdout (not full k-fold) —
a deliberate CPU-budget tradeoff, documented. Same model, same epochs, same
code path for both; only the split differs.

Run:  .venv/bin/python src/spatial_cv.py --epochs 3
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from train import DATA_DIR, IMAGENET_TRANSFORM, OUT_DIR, build_model, pick_device

BLOCK_M = 50_000  # 50 km grid blocks
VAL_FRACTION = 0.2
SEED = 42
BATCH_SIZE = 128  # part of the identical-training contract for both splits
PLOT_ROWS, PLOT_COLS = 2, 3  # zones plotted = ROWS*COLS largest


def load_coords() -> dict:
    """filename -> (epsg, easting, northing)"""
    coords = {}
    with open(DATA_DIR / "coords.csv") as f:
        for row in csv.DictReader(f):
            coords[row["filename"]] = (int(row["epsg"]),
                                       float(row["easting"]),
                                       float(row["northing"]))
    return coords


def spatial_split(ds, coords):
    """Assign each patch to a 50km block; hold out whole blocks for val.

    LESSON: the unit we shuffle is the BLOCK, not the image. Two images in
    the same block can never end up on opposite sides of the split — that's
    the whole trick.
    """
    block_members = defaultdict(list)
    for i, (path, _) in enumerate(ds.samples):
        epsg, e, n = coords[Path(path).name]
        block = (epsg, int(e // BLOCK_M), int(n // BLOCK_M))
        block_members[block].append(i)

    rng = np.random.default_rng(SEED)
    blocks = list(block_members)
    rng.shuffle(blocks)
    target_val = int(len(ds) * VAL_FRACTION)
    val_idx, n_val_blocks = [], 0
    for b in blocks:
        if len(val_idx) >= target_val:
            break
        n_val_blocks += 1
        val_idx.extend(block_members[b])
    train_idx = sorted(set(range(len(ds))) - set(val_idx))
    print(f"  spatial: {len(block_members)} blocks total, "
          f"{n_val_blocks} held out -> {len(val_idx)} val images")
    return train_idx, val_idx


def random_split_idx(n):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    n_val = int(n * VAL_FRACTION)
    return order[n_val:].tolist(), order[:n_val].tolist()


def train_eval(ds, train_idx, val_idx, device, epochs, lr, tag):
    """Identical training for both splits — only the indices differ."""
    model = build_model(num_classes=len(ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
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
        print(f"  [{tag}] epoch {epoch}/{epochs} ({time.time() - t0:.0f}s)")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            preds.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return accuracy_score(targets, preds), cohen_kappa_score(targets, preds)


def plot_blocks(ds, coords, val_idx):
    """Scatter the patches of the largest UTM zones, colored by split."""
    val_set = set(val_idx)
    by_zone = defaultdict(list)
    for i, (path, _) in enumerate(ds.samples):
        epsg, e, n = coords[Path(path).name]
        by_zone[epsg].append((e, n, i in val_set))
    zones = sorted(by_zone, key=lambda z: -len(by_zone[z]))[:PLOT_ROWS * PLOT_COLS]
    fig, axes = plt.subplots(PLOT_ROWS, PLOT_COLS, figsize=(14, 8))
    for ax, z in zip(axes.flat, zones):
        pts = np.array([(e, n) for e, n, _ in by_zone[z]])
        isval = np.array([v for _, _, v in by_zone[z]])
        ax.scatter(*pts[~isval].T / 1000, s=2, c="#1976d2", label="train")
        ax.scatter(*pts[isval].T / 1000, s=2, c="#ef6c00", label="val")
        ax.set_title(f"EPSG:{z} ({len(pts)} patches)", fontsize=9)
        ax.set_xlabel("easting km")
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(markerscale=4)
    fig.suptitle("Spatial split — whole 50km blocks held out (orange)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_blocks.png", dpi=120)
    print(f"  saved {OUT_DIR / 'spatial_blocks.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    ds = datasets.EuroSAT(root=DATA_DIR, download=False,
                          transform=IMAGENET_TRANSFORM)
    coords = load_coords()

    print("Random split:")
    tr_r, va_r = random_split_idx(len(ds))
    acc_r, kap_r = train_eval(ds, tr_r, va_r, device, args.epochs, args.lr, "random")
    print(f"  random:  acc {acc_r:.4f}  kappa {kap_r:.4f}")

    print("Spatial split:")
    tr_s, va_s = spatial_split(ds, coords)
    acc_s, kap_s = train_eval(ds, tr_s, va_s, device, args.epochs, args.lr, "spatial")
    print(f"  spatial: acc {acc_s:.4f}  kappa {kap_s:.4f}")

    plot_blocks(ds, coords, va_s)
    result = {
        "random": {"accuracy": round(acc_r, 4), "kappa": round(kap_r, 4)},
        "spatial": {"accuracy": round(acc_s, 4), "kappa": round(kap_s, 4)},
        "gap_accuracy": round(acc_r - acc_s, 4),
        "gap_kappa": round(kap_r - kap_s, 4),
        "epochs": args.epochs, "block_km": BLOCK_M // 1000,
        "note": "single holdout per condition (CPU budget), identical training",
    }
    (OUT_DIR / "spatial_gap.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
