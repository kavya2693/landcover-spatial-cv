"""Step 1 — Look at the data before touching any model.

LESSON: Every ML project starts here. An image is just a grid of numbers:
a 64x64 colour satellite patch is a 3x64x64 box of values (3 = red/green/blue
channels). "Classification" means mapping that box of numbers to one of 10
labels (Forest, River, Industrial, ...). If you understand the data, half
the interview questions answer themselves.

Run:  .venv/bin/python src/explore_data.py
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    # download=True fetches the dataset the first time (~90 MB), then reuses it
    ds = datasets.EuroSAT(root=DATA_DIR, download=True)

    print(f"Total images: {len(ds)}")
    print(f"Classes ({len(ds.classes)}): {ds.classes}")

    # LESSON: class balance matters. If 90% of images were Forest, a lazy
    # model could say "Forest" every time and score 90% accuracy while
    # learning nothing. That is why we will also use Cohen's kappa later.
    counts = Counter(label for _, label in ds.samples)
    print("\nImages per class:")
    for idx, name in enumerate(ds.classes):
        print(f"  {name:<22}{counts[idx]}")

    # Save one example image per class so we can SEE the problem.
    # ds.samples already holds (path, label) pairs — pick the first path per
    # class and decode only those 10 JPEGs instead of iterating the dataset.
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    seen = {}
    for path, label in ds.samples:
        if label not in seen:
            seen[label] = path
        if len(seen) == len(ds.classes):
            break
    for ax, (label, path) in zip(axes.flat, sorted(seen.items())):
        ax.imshow(Image.open(path))
        ax.set_title(ds.classes[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("EuroSAT — one example per class (64x64 Sentinel-2 patches)")
    fig.tight_layout()
    out = OUT_DIR / "class_examples.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved sample grid -> {out}")


if __name__ == "__main__":
    main()
