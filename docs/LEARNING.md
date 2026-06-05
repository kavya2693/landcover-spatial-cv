# LEARNING.md — Understand This Project From Zero

A layered guide: read Layer 1 today, Layer 2 after the first training run, Layer 3 before touching Phase B, Layer 4 before any interview. Each layer is anchored to *this* project — real numbers, real files.

---

## Layer 1 — Intuition (the 7 ideas everything else builds on)

### 1. An image is just numbers
Each EuroSAT patch is a 64×64 grid of pixels, each pixel holding 3 numbers (red, green, blue intensity). So one image = a 3×64×64 box of numbers. That's all the computer ever sees.

### 2. Classification = numbers in, label out
Our task: take that box of numbers and output one of 10 labels (Forest, River, Industrial...). A "model" is just a giant mathematical function with millions of adjustable knobs ("weights") that maps the numbers to a label.

### 3. Learning = guess → measure → adjust
The model starts by guessing. A **loss function** scores how wrong it was. An **optimizer** then nudges every knob slightly in the direction that would have made it less wrong. Do this thousands of times and the guesses become good. One full pass over the training data = one **epoch**. This loop is in `src/train.py` — find the three lines commented GUESS/MEASURE/ADJUST.

### 4. CNNs fit images because patterns are local
A **Convolutional Neural Network** slides small filters (like 3×3 magnifying glasses) across the image. Early filters detect edges and colours; deeper layers combine those into textures, then shapes, then whole concepts ("grid of streets" → Residential). This matches how images actually work: a river pixel's meaning depends on its neighbours, not on a pixel in the far corner.

### 5. Transfer learning = don't start from zero
**ResNet18** is a CNN already trained on 1.2 million everyday photos (ImageNet). Its filters already detect edges, textures, shapes — those are useful for satellite images too. We **freeze** all of it and replace only the final layer (1000 ImageNet classes → our 10). Only that tiny new layer learns. That's why training takes minutes on a laptop CPU instead of days on a GPU.

### 6. Hold out data to measure honestly
We train on 80% of images and validate on 20% the model never saw. Performance on *unseen* data is the only number that matters — anyone can memorize the training set.

### 7. Metrics can lie — this project's signature theme
- **Accuracy** can flatter: with class imbalance (Pasture 2,000 vs Forest 3,000), always-guess-the-big-class scores well while learning little.
- **Cohen's kappa** corrects for lucky/chance agreement → more honest.
- **Random splits** flatter satellite models: two patches from the same farm can land in train AND validation — the model "recognizes the farm" instead of "understanding cropland". That's **spatial autocorrelation leakage**, and fixing it with **spatial cross-validation** is Phase B and the most interview-worthy part of this project.

---

## Layer 2 — Mechanics (what each piece in the code actually does)

| Piece | In our code | What it does | Why this choice |
|---|---|---|---|
| `ToTensor` + `Normalize` | `train.py` transforms | Scales pixels and shifts them to the range the pretrained net expects | Must match ImageNet's preprocessing or the frozen features misfire |
| `resnet18(weights=DEFAULT)` | `build_model()` | Loads the pretrained CNN | Small, fast, strong baseline |
| `requires_grad = False` | `build_model()` | Freezes backbone weights | Nothing to compute gradients for → fast CPU training |
| `nn.Linear(512, 10)` | `model.fc` | The new trainable head | 512 features in → 10 class scores out |
| `CrossEntropyLoss` | `criterion` | Loss for multi-class problems: "how surprised by the true label" | The standard; punishes confident wrong answers hardest |
| `Adam, lr=1e-3` | `optimizer` | Adjusts the head's weights each batch | Adaptive step sizes; 1e-3 is a sane default |
| `batch_size=128` | DataLoader | Images processed per step | Bigger = faster but more memory; 128 fits comfortably |
| `random_split` seed 42 | split | Reproducible 80/20 split | Same split every run → comparable results |
| `model.eval()` + `no_grad()` | val loop | Switches off training behaviours, skips gradient bookkeeping | Evaluation must not learn or waste compute |

**Key vocabulary you now own:** epoch, batch, loss, gradient, optimizer, learning rate, backbone, head, freezing, normalization, train/val split, inference.

---

## Layer 3 — Honesty (the evaluation story that sets this project apart)

1. **Confusion matrix** (`outputs/confusion_matrix.png`): rows = true class, columns = predicted. Off-diagonal cells show *which* classes get confused (expect PermanentCrop ↔ AnnualCrop ↔ HerbaceousVegetation). One number can't tell you that.
2. **Cohen's kappa**: (observed agreement − chance agreement) / (1 − chance agreement). Kappa = 0 means "no better than chance", 1 means perfect. It's the standard in remote sensing precisely because land-cover classes are imbalanced.
3. **Spatial cross-validation** (Phase B): instead of splitting patches randomly, split by *geographic region* so the model is validated on places it has never seen. Scores typically DROP — and that drop is the honest gap this project is designed to measure and report. Saying "my random-split accuracy was inflated and here is by how much" is what makes a portfolio project sound senior.

---

## Layer 4 — Articulation (sound like you built it, because you did)

Your 30-second project pitch:

> "I built a land-cover classifier on EuroSAT — 27,000 Sentinel-2 satellite patches, 10 classes. I fine-tuned a frozen ResNet18 with transfer learning, which trains in minutes on a laptop CPU. I reported Cohen's kappa alongside accuracy because the classes are imbalanced, and a confusion matrix to show which crop classes get confused. The interesting part: random train/val splits overstate performance on satellite data because nearby patches leak across the split, so I measured the gap against spatial cross-validation."

Then go to `docs/INTERVIEW_QA.md` and practice the full question bank.

---

## Layer 5 — Phase B: Spatial Cross-Validation (the signature finding)

**The suspicion.** Tobler's first law of geography: *near things are more alike than far things*. EuroSAT cut big satellite scenes into 64×64 patches — so many patches are next-door neighbours of the same farm, forest, or town. A **random** split scatters those neighbours across train AND validation. The model can then score well by *recognizing the place* (nearly identical pixels it trained on) rather than *understanding the land-cover concept*. The 85.1% might be partly an illusion.

**Recovering locations.** Our RGB jpegs carry no location — but EuroSAT's multispectral GeoTIFFs are *georeferenced*: TIFF metadata tags store a **tiepoint** (the map coordinate of the corner pixel), a **pixel scale** (10m for Sentinel-2), and an **EPSG code** naming the UTM zone. `src/extract_coords.py` reads those three tags for all 27,000 patches straight out of the zip (we never even extracted it — and deleted the 2GB after saving a 1MB `coords.csv`). Industry would use `rasterio`; it has no Python 3.14 wheel yet, so we parsed the tags directly with `tifffile` — worth mentioning in an interview, it shows you know what's *inside* a GeoTIFF.

**The experiment** (`src/spatial_cv.py`). Group patches into **50 km grid blocks** (UTM zone + easting/northing cell). Train the identical model twice:
- **Random split** — 20% of *images* held out (neighbours leak across)
- **Spatial split** — 20% of *blocks* held out whole (no neighbour can leak)
Same architecture, epochs, learning rate, code path. Only the split differs — so any score difference is pure leakage.

**Honest footnote:** we ran one holdout per condition rather than full 5-fold CV — a deliberate CPU-budget tradeoff, stated in the output JSON. Full GroupKFold is the gold standard; say so if asked.

**The measured result — a null finding, honestly reported.** Random 85.2% vs spatial 85.6%: **no gap** (−0.4% is within run-to-run noise). The leakage we hypothesized did not appear in this configuration. Why? The leading hypothesis: our backbone is *frozen* — only a 5,130-parameter linear head trains, and memorizing specific places needs capacity the head doesn't have. Spatial leakage bites hardest when the whole network can overfit. This sets up Phase C's sharpest question: *does fine-tuning all 11M parameters reopen the gap?*

**Why a null result is interview GOLD.** Weak candidates report only flattering numbers. Strong ones build the apparatus, run the controlled experiment (identical training, only the split differs), report whatever comes out, and propose the follow-up. Your sentence:

> "I hypothesized spatial leakage, recovered all 27k patch coordinates from GeoTIFF metadata, held out whole 50km blocks — and found no gap with a frozen backbone. My hypothesis is that leakage requires memorization capacity, which is exactly what Phase C tests by unfreezing the network."

## What's next (the project's remaining phases)
- **Phase B**: ✅ done — see the 🧪 tab for your measured gap
- **Phase C**: fine-tune vs frozen; CNN vs transformer; per-class IoU
- **Phase D**: publish + mock-interview gate (12+/14 on the 🎯 Quiz tab)
