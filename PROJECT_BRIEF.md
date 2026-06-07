# Project Brief — Land-Cover Classification (Project 1 of 3)

**Updated:** 2026-06-05 · supersedes the Project 1 section of `Portfolio_Project_Briefs.docx`
**Owner:** Srikavya · **AI pair:** Nova (builds + tutors in parallel)

---

## The dual mission (what changed in this revision)

This project has **two tracks with equal weight**. The original brief covered only Track 1.

| Track | Goal | Why |
|---|---|---|
| 🔨 **BUILD** | A portfolio-grade repo demonstrating remote-sensing ML competence | Gets the interview |
| 🎓 **LEARN** | Srikavya can explain every artifact in this repo **unaided** | Passes the interview |

**Operating rule:** no build step ships without its learning step. Every phase ends with both a technical deliverable AND a tutoring deliverable (LEARNING.md section, tutor-KB entries, and a passed quiz checkpoint). A repo you can't explain is a liability in an interview, not an asset.

---

## Problem statement (unchanged)

Given multispectral satellite imagery, classify land cover into 10 types with **quantified, spatially honest accuracy** — and show whether a transformer matches/beats a CNN under proper spatial validation rather than an over-optimistic random split.

---

## Goal conditions per phase

Written as verifiable end-states: each condition is binary-checkable — a file exists, a number is logged, or Srikavya answers unaided. "Done" means **all** conditions in the phase pass, both tracks.

### ✅ Phase A — Baseline (COMPLETE 2026-06-04)

**Build conditions — all met:**
- [x] EuroSAT (27,000 patches, 10 classes) loads via reproducible script → `src/explore_data.py`
- [x] ResNet18 frozen-backbone baseline trains on laptop CPU → `src/train.py`
- [x] Metrics logged: **val accuracy 0.8509, Cohen's kappa 0.834** → `outputs/metrics.json`
- [x] Confusion matrix saved; top confusions identified (River→Highway 81, PermanentCrop→HerbaceousVegetation 49)
- [x] Repo: git history, README with results table, requirements.txt, .gitignore

**Learn conditions — met:**
- [x] `docs/LEARNING.md` — 4-layer curriculum (intuition → mechanics → honesty → articulation)
- [x] `docs/INTERVIEW_QA.md` — 35 anchored questions
- [x] `docs/visual_guide.html` — 6 interactive sections + AI tutor sidebar (instant KB + live Claude answers, concept tabs, refresh-persistent)
- [x] Concepts covered live: labels/ImageFolder, supervised learning, train/val splits, "unseen" data

### ✅ Phase B — Spatial cross-validation (COMPLETE 2026-06-05 · finding: honest null)

**Build conditions — all met:**
- [x] `src/spatial_cv.py` implements geographically-separated folds (50km blocks; 866 blocks, 147 held out; coordinates recovered from GeoTIFF tags via `src/extract_coords.py`)
- [x] Same model evaluated under random vs spatial folds, same seed policy (identical training, only indices differ)
- [x] **Gap quantified and logged**: `outputs/spatial_gap.json` — random 85.2% / spatial 85.6% / gap −0.4% (within noise)
- [x] README results table gains spatial CV row + plain-language finding paragraph

**The measured finding:** NO leakage gap in the frozen-backbone configuration — hypothesis: the 5,130-param head lacks memorization capacity. Phase C tests whether full fine-tuning reopens the gap. (Original brief assumed a positive gap; the apparatus was built, the experiment run, the null reported honestly.)

**Learn conditions:**
- [x] LEARNING.md Layer 5 (Tobler's law → leakage → fold design → null-result interpretation)
- [x] Tutor KB entries: spatial autocorrelation, k-fold CV, GeoTIFF coordinates (11 total)
- [ ] **Quiz gate (Srikavya's move):** score 12+/14 on the 🎯 Quiz tab, and explain unaided: (1) why random splits *can* flatter satellite models, (2) what her measured gap was and why it was null, (3) how the folds were built

### ⬜ Phase C — Scale & compare

**Build conditions:**
- [x] Fine-tuned (unfrozen) ResNet vs frozen baseline — delta reported (+11.2 pts: 85.2%→96.4% random, 85.6%→95.8% spatial; gap −0.4%→+0.7%, small leakage signal appeared with capacity — `outputs/finetune_gap.json`)
- [ ] One transformer-family model (e.g. ViT via torchgeo) benchmarked under identical spatial folds
- [ ] Per-class IoU/F1 table in README; CNN-vs-transformer verdict stated with numbers
- [ ] Constraint: stays trainable on available hardware (CPU/Colab free tier) — document wall-clock

**Learn conditions:**
- [ ] LEARNING.md: fine-tuning vs feature extraction; attention/ViT in one intuition-first section
- [ ] Quiz gate: explain the frozen→fine-tuned→transformer progression and when each wins

### ⬜ Phase D — Publish

**Build conditions:**
- [ ] GitHub repo public, metrics-first README leading with the spatial-CV finding
- [ ] Short write-up (blog post or README essay): "My 85% was lying to me — by exactly N points"
- [ ] Clean `git log` telling the project story

**Learn conditions:**
- [ ] **Mock interview gate:** 30-second pitch + 10 random questions from INTERVIEW_QA.md answered unaided, including 2 gotchas
- [ ] Srikavya can whiteboard the full pipeline from satellite to prediction

---

## Constraints (unchanged + learned)

- **Cost:** free data, free compute only. No API keys (tutor uses Claude subscription via PAI).
- **Hardware reality (measured):** macOS 13.4 → no MPS in modern PyTorch → CPU-only; ~140s/epoch frozen-backbone. Disk ~17GB free → RGB EuroSAT only locally; full Sentinel-2 tiles via Colab/streaming in Phase C.
- **Scope guard:** no edge deployment, no STAC catalogues, no Projects 2/3 work inside this repo.

## Tooling that exists for the LEARN track

| Tool | Run | Purpose |
|---|---|---|
| Visual guide + AI tutor | `.venv/bin/python src/tutor_server.py` → http://localhost:8787/visual_guide.html | Ask anything; visual animated answers; concept tabs |
| Curriculum | `docs/LEARNING.md` | Read in layers, in order |
| Question bank | `docs/INTERVIEW_QA.md` | Practice out loud; quiz gates draw from here |

## Definition of project-done

> The repo is public with the spatial-CV finding front and center, **and** Srikavya passes the Phase D mock-interview gate without help. Both — or it's not done.
