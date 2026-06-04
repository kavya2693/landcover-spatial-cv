# Land-Cover Classification from Sentinel-2 Satellite Imagery

Classifying 27,000 EuroSAT satellite patches into 10 land-cover classes (Forest, River, Industrial, cropland types, ...) with **spatially honest evaluation** — measuring how much a standard random train/val split overstates performance versus spatial cross-validation.

## Results (Phase A baseline — updated as phases complete)

| Model | Split | Accuracy | Cohen's kappa |
|---|---|---|---|
| ResNet18 (frozen backbone, transfer learning, 3 epochs, CPU) | random 80/20 | **85.1%** | **0.834** |

Top confusions: River→Highway (81) — both linear features; PermanentCrop→HerbaceousVegetation (49) — similar vegetation texture at 10 m resolution.

![Confusion matrix](outputs/confusion_matrix.png)

## Why this project
Land-cover classification is the foundational task of Earth observation. This repo demonstrates:
- **Vision-model fluency** — CNN transfer learning, later CNN-vs-transformer benchmarking
- **Rigorous geospatial validation** — Cohen's kappa, per-class IoU, and spatial CV to eliminate spatial-autocorrelation leakage
- **Reproducibility** — seeded splits, pinned requirements, metrics-first reporting

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/explore_data.py   # downloads EuroSAT (~90 MB), prints class stats
python src/train.py --epochs 3
```

## Roadmap
- [x] Phase A — ResNet18 baseline on EuroSAT, accuracy + kappa + confusion matrix
- [ ] Phase B — spatial cross-validation; quantify the random-vs-spatial gap
- [ ] Phase C — full Sentinel-2 tiles; CNN vs transformer; per-class IoU
- [ ] Phase D — write-up framed around the spatial-CV finding

## Repo guide
- `src/explore_data.py` — dataset download + class distribution + sample grid
- `src/train.py` — transfer-learning baseline (heavily commented for learning)
- `docs/LEARNING.md` — layered beginner-to-interview explanation of every concept used
- `docs/INTERVIEW_QA.md` — 30 interview questions with model answers

Data: [EuroSAT](https://github.com/phelber/EuroSAT) (Sentinel-2, CC-BY-4.0). Free data, free compute — trains on a laptop CPU in minutes.
