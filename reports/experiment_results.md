# MedReason++ — Experiment Results Log

This file records **manually verified experiment results** for the
MedReason++ pneumonia classification project.

Rules for this file:
- Entries are added by the researcher after reviewing outputs; they are
  **not** generated automatically by training scripts.
- Each entry clearly states the training conditions, hardware, and any
  limitations on reliability.
- Do **not** use the values in this file as model predictions or training
  labels.
- Do **not** treat these as independently clinically verified results.
- Superseded results are kept with a `[SUPERSEDED]` marker rather than
  deleted, so the history is preserved.

Auto-generated per-run metric files live under `processed/<model>/test_metrics.json`.
Those files are overwritten on each run; this log is the persistent record.

---

## Dataset

**Kaggle Chest X-Ray Images (Pneumonia)**
- Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Population: **pediatric** chest X-rays
- Original split used for all experiments:

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train (original) | 1,341 | 3,875 | 5,216 |
| Validation (original) | 8 | 8 | 16 |
| Test (untouched) | 234 | 390 | 624 |

**Important:** The original 8-image validation set is too small for reliable
model selection. All experiments below use a **stratified 15% split** of the
training data as the working validation set (seed=42), producing:

| Working split | NORMAL | PNEUMONIA | Total |
|--------------|-------:|----------:|------:|
| Train | 1,140 | 3,294 | 4,434 |
| Val (derived) | 201 | 581 | 782 |
| Test (untouched) | 234 | 390 | **624** |

The 624-image test set was **never used** during training, model selection,
or threshold tuning.

**Class mapping:** NORMAL = 0 (negative class), PNEUMONIA = 1 (positive class).

---

## Models

### Experiment 1 — ResNet-18 Baseline (Phase 6)

| Field | Value |
|-------|-------|
| Model | ResNet-18 pretrained on ImageNet-1k (torchvision) |
| Parameters | 11.7M |
| Hardware | CPU (laptop) |
| Status | **Preliminary — CPU-trained** |
| Epochs trained | 3 |
| Training time | ~29.6 minutes |
| Script | `src/train_pneumonia_baseline.py` |
| Checkpoint | `processed/phase6/best_model.pt` |
| Threshold | 0.50 (default) |

**Test-set results (624 images):**

| Metric | Value |
|--------|------:|
| Accuracy | 0.8381 |
| Precision | 0.7955 |
| Recall (sensitivity) | 0.9974 |
| Specificity | 0.5726 |
| F1-score | 0.8851 |
| ROC-AUC | 0.9610 |

Confusion matrix:

| | Pred NORMAL | Pred PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (234) | 134 TN | 100 FP |
| **Actual PNEUMONIA** (390) | 1 FN | 389 TP |

---

### Experiment 2 — ResNet-18 Extended (Phase 6b)

| Field | Value |
|-------|-------|
| Model | ResNet-18 warm-started from Experiment 1 epoch-3 checkpoint |
| Hardware | CPU (laptop) |
| Status | **Preliminary — CPU-trained** |
| Epochs trained | 4 (early stopping on val ROC-AUC, patience=3) |
| Training time | ~41 minutes |
| Script | `src/finetune_pneumonia.py` |
| Checkpoint | `processed/phase6b/best_model.pt` |
| Threshold | 0.50 (default) |

**Test-set results (624 images) at threshold 0.50:**

| Metric | Value |
|--------|------:|
| Accuracy | 0.8878 |
| Precision | 0.8524 |
| Recall (sensitivity) | 0.9923 |
| Specificity | 0.7137 |
| F1-score | 0.9171 |
| ROC-AUC | 0.9679 |

Confusion matrix:

| | Pred NORMAL | Pred PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (234) | 167 TN | 67 FP |
| **Actual PNEUMONIA** (390) | 3 FN | 387 TP |

---

### Experiment 3 — ConvNeXt-Tiny (Phase 7a)

| Field | Value |
|-------|-------|
| Model | ConvNeXt-Tiny pretrained on ImageNet-1k (torchvision) |
| Parameters | 27.8M |
| Hardware | **GPU — Kaggle Tesla T4** |
| Status | **Complete — GPU-trained** |
| Epochs trained | 8 (early stopping at epoch 8, val AUC peaked at epoch 5 = 0.9992) |
| Script | `src/train_convnext.py` / `kaggle_train.py` |
| Threshold | 0.50 (default) |

**Validation history:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|------:|-----------:|----------:|---------:|--------:|--------:|
| 1 | 0.1632 | 0.9312 | 0.0961 | 0.9463 | 0.9963 |
| 2 | 0.0605 | 0.9765 | 0.0512 | 0.9783 | 0.9985 |
| 3 | 0.0475 | 0.9829 | 0.0550 | 0.9783 | 0.9984 |
| 4 | 0.0450 | 0.9842 | 0.0540 | 0.9770 | 0.9989 |
| 5 | 0.0312 | 0.9901 | 0.0367 | 0.9859 | **0.9992** ← best |
| 6 | 0.0206 | 0.9930 | 0.0592 | 0.9719 | 0.9990 |
| 7 | 0.0155 | 0.9944 | 0.0508 | 0.9795 | 0.9990 |
| 8 | 0.0140 | 0.9944 | 0.0404 | 0.9898 | 0.9991 |

**Test-set results (624 images) at threshold 0.50:**

| Metric | Value |
|--------|------:|
| Accuracy | 0.8574 |
| Precision | — |
| Recall (sensitivity) | 0.9949 |
| Specificity | 0.6282 |
| F1-score | 0.8971 |
| ROC-AUC | 0.9713 |

Confusion matrix:

| | Pred NORMAL | Pred PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (234) | 147 TN | 87 FP |
| **Actual PNEUMONIA** (390) | 2 FN | 388 TP |

---

### Experiment 4 — CXformer-S (Phase 7b)

| Field | Value |
|-------|-------|
| Model | CXformer-S (m42-health/CXformer-small, CC BY-NC-4.0, research only) |
| Parameters | 22.1M total, 0.001M trainable (frozen backbone, linear probing — head only) |
| Input size | 518×518 (CXformer's own AutoImageProcessor) |
| Hardware | **GPU — Kaggle Tesla T4** |
| Status | **Complete — GPU-trained** |
| Epochs trained | 10 (all 10 ran, val AUC improved every epoch — still improving at epoch 10) |
| Script | `kaggle_train.py` |
| Threshold | 0.50 (default) |

**Validation history:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|------:|-----------:|----------:|---------:|--------:|--------:|
| 1 | 0.5504 | 0.6673 | 0.3565 | 0.8235 | 0.9445 |
| 2 | 0.2865 | 0.8778 | 0.2475 | 0.8721 | 0.9768 |
| 3 | 0.2193 | 0.9053 | 0.2035 | 0.8990 | 0.9841 |
| 4 | 0.1825 | 0.9222 | 0.1794 | 0.9092 | 0.9872 |
| 5 | 0.1647 | 0.9328 | 0.1644 | 0.9258 | 0.9889 |
| 6 | 0.1549 | 0.9371 | 0.1561 | 0.9246 | 0.9897 |
| 7 | 0.1473 | 0.9432 | 0.1500 | 0.9297 | 0.9902 |
| 8 | 0.1398 | 0.9452 | 0.1467 | 0.9297 | 0.9905 |
| 9 | 0.1386 | 0.9488 | 0.1443 | 0.9309 | 0.9907 |
| 10 | 0.1360 | 0.9472 | 0.1427 | 0.9309 | **0.9908** ← best |

> Note: Val AUC improved every epoch — more epochs may improve further.

**Test-set results (624 images) at threshold 0.50:**

| Metric | Value |
|--------|------:|
| Accuracy | **0.9103** |
| Recall (sensitivity) | 0.9590 |
| Specificity | **0.8291** |
| F1-score | **0.9303** |
| ROC-AUC | 0.9702 |

Confusion matrix:

| | Pred NORMAL | Pred PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (234) | 194 TN | 40 FP |
| **Actual PNEUMONIA** (390) | 16 FN | 374 TP |

---

## Comparison Table (all experiments)

| Model | Hardware | Epochs | Accuracy | Recall | Specificity | F1 | ROC-AUC |
|-------|----------|-------:|---------:|-------:|------------:|---:|--------:|
| ResNet-18 (baseline) | CPU | 3 | 83.8% | 99.7% | 57.3% | 88.5% | 96.1% |
| ResNet-18 (extended) | CPU | 4 | 88.8% | 99.2% | 71.4% | 91.7% | 96.8% |
| ConvNeXt-Tiny | **T4 GPU** | 8 | 85.7% | 99.5% | 62.8% | 89.7% | 97.1% |
| **CXformer-S** | **T4 GPU** | 10 | **91.0%** | 95.9% | **82.9%** | **93.0%** | 97.0% |

## Comparison Table — All Experiments (test set, 624 images, threshold=0.50)

| Model | Hardware | Epochs | Accuracy | Recall | Specificity | F1 | ROC-AUC |
|-------|----------|-------:|---------:|-------:|------------:|---:|--------:|
| ResNet-18 (baseline) | CPU | 3 | 83.8% | 99.7% | 57.3% | 88.5% | 96.1% |
| ResNet-18 (extended) | CPU | 4 | 88.8% | 99.2% | 71.4% | 91.7% | 96.8% |
| ConvNeXt-Tiny | T4 GPU | 8 | 85.7% | 99.5% | 62.8% | 89.7% | 97.1% |
| CXformer-S | T4 GPU | 10 | **91.0%** | 95.9% | **82.9%** | 93.0% | 97.0% |
| **ConvNeXt + CXformer Hybrid** | **T4 GPU** | **8** | **90.7%** | **99.5%** | **76.1%** | **93.1%** | **98.8%** |

### Key findings

**Hybrid vs individual models:**

| Metric | CXformer-S (best individual) | Hybrid | Change |
|--------|-----------------------------:|-------:|--------|
| Accuracy | 91.0% | 90.7% | ↓ 0.3% (negligible) |
| Recall | 95.9% | **99.5%** | ↑ **+3.6%** |
| Specificity | 82.9% | 76.1% | ↓ 6.8% |
| F1 | 93.0% | **93.1%** | ↑ 0.1% |
| ROC-AUC | 97.0% | **98.8%** | ↑ **+1.8%** |

**What this means:**

1. **ROC-AUC 98.8% is the highest of all models** — the hybrid has the best overall discriminative ability across all thresholds. This is the most important metric for a screening tool.

2. **Recall 99.5% — only 2 missed pneumonia cases out of 390** — the hybrid almost never misses a sick patient. CXformer-S alone missed 16.

3. **The hybrid achieves near-identical accuracy to CXformer-S (90.7% vs 91.0%)** while improving recall by 3.6 percentage points and ROC-AUC by 1.8 points.

4. **Specificity trade-off is expected** — higher recall naturally comes at the cost of more false positives (56 FP vs 40 FP). For a medical screening tool, this is the correct trade-off.

5. **The fusion is doing what it was designed to do:** ConvNeXt contributes strong local feature detection (catching nearly all pneumonia), CXformer contributes high-level chest-specific understanding (better normal image recognition).

> **Note on population:** All experiments use **pediatric** chest X-rays.
> Results must not be generalised to adult populations without additional validation.

> **Not clinical ground truth:** Research experiment only. NOT a validated clinical diagnostic tool.

---

## How to Update This File

After completing further experiments (Grad-CAM, threshold tuning, etc.):
1. Add a new entry below Experiment 5
2. Update the comparison table
3. Keep superseded results with their original values

---

_Last updated: Hybrid ConvNeXt-Tiny + CXformer-S GPU results recorded from Kaggle T4._

> **Note on population:** All experiments use pediatric chest X-rays.
> Results should not be generalised to adult populations without additional validation.

> **Not clinical ground truth:** All labels are radiologist-annotated lung
> opacity findings (Kermany et al.). This is a research experiment, not a
> validated clinical diagnostic tool.

---

## How to Update This File

After completing a GPU training run:
1. Run `python src/evaluate_checkpoint.py --model_type convnext ...`
2. Copy the printed metrics into the relevant entry above.
3. Mark the CPU entry as `[SUPERSEDED by GPU run]`.
4. Add a new entry for the GPU result with updated hardware, epoch count, and metrics.

---

### Experiment 5 — ConvNeXt-Tiny + CXformer-S Hybrid (Phase 8)

| Field | Value |
|-------|-------|
| Model | Feature-level fusion: ConvNeXt-Tiny (768-dim) + CXformer-S (384-dim) |
| Architecture | Both encoders frozen → projection layers (768→256, 384→256) → concat 512-dim → classifier |
| Total parameters | 50.3M |
| Trainable parameters | 0.428M (projection layers + fusion classifier only) |
| Frozen parameters | 49.9M (both pretrained encoders) |
| Hardware | **GPU — Kaggle Tesla T4** |
| Status | **Complete — GPU-trained** |
| Epochs trained | 8 (early stopping, patience=3, best at epoch 5 val AUC=0.9985) |
| Script | `kaggle_train.py` (hybrid section) |
| Threshold | 0.50 (default) |

**Fusion architecture:**
```
ConvNeXt-Tiny 768-dim ─► Linear(768→256) + LayerNorm + GELU ─┐
                                                               ├─ concat 512 ─► Linear(512→256) ─► GELU ─► Dropout(0.3) ─► Linear(256→2)
CXformer-S    384-dim ─► Linear(384→256) + LayerNorm + GELU ─┘
```

**Validation history:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val AUC |
|------:|-----------:|----------:|---------:|--------:|--------:|
| 1 | 0.1450 | 0.9463 | 0.0843 | 0.9604 | 0.9977 |
| 2 | 0.0573 | 0.9797 | 0.0675 | 0.9693 | 0.9975 |
| 3 | 0.0411 | 0.9847 | 0.0707 | 0.9655 | 0.9982 |
| 4 | 0.0315 | 0.9871 | 0.0677 | 0.9668 | 0.9983 |
| 5 | 0.0319 | 0.9892 | 0.0554 | 0.9770 | **0.9985** ← best |
| 6 | 0.0198 | 0.9939 | 0.0601 | 0.9757 | 0.9983 |
| 7 | 0.0216 | 0.9930 | 0.0704 | 0.9795 | 0.9982 |
| 8 | 0.0108 | 0.9962 | 0.0607 | 0.9783 | 0.9984 |

**Test-set results (624 images) at threshold 0.50:**

| Metric | Value |
|--------|------:|
| Accuracy | 0.9071 |
| Precision | 0.8739 |
| Recall (sensitivity) | **0.9949** |
| Specificity | 0.7607 |
| F1-score | **0.9305** |
| ROC-AUC | **0.9876** |

Confusion matrix:

| | Pred NORMAL | Pred PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (234) | 178 TN | 56 FP |
| **Actual PNEUMONIA** (390) | 2 FN | 388 TP |
