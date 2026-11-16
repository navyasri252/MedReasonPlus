# Phase 6b — Extended Fine-Tuning Comparison Report

> **⚠ CLINICAL DISCLAIMER:** Research prototype. NOT validated for clinical use.

---

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| `warm_start_from` | C:\Users\navya\AppData\Local\Temp\pytest-of-navya\pytest-32\test_test_set_not_accessed_bef0\fake_checkpoint.pt |
| `dataset_root` | C:\Users\navya\AppData\Local\Temp\pytest-of-navya\pytest-32\test_test_set_not_accessed_bef0\chest_xray |
| `model` | ResNet18 (warm-start from epoch-3) |
| `max_epochs` | 1 |
| `patience` | 1 |
| `lr` | 0.0001 |
| `batch_size` | 4 |
| `seed` | 42 |
| `device` | cpu |
| `total_time_s` | 3.1 |
| `epochs_trained` | 1 |
| `recommended_threshold` | 0.1 |

---

## 2. Baseline Reference (Phase 6, 3 epochs, threshold=0.5)

| Metric | Value |
|--------|------:|
| Accuracy | 0.8381 |
| Precision | 0.7955 |
| Recall | 0.9974 |
| Specificity | 0.5726 |
| F1 | 0.8851 |
| Roc auc | 0.961 |

Confusion matrix: TN=134, FP=100, FN=1, TP=389

---

## 3. Extended Training History (warm-started from epoch-3 checkpoint)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val ROC-AUC | Val Recall | Val Spec | Note |
|------:|-----------:|----------:|---------:|--------:|------------:|-----------:|---------:|------|
| 1 | 0.7763 | 0.3333 | 0.7053 | 0.5000 | 0.5000 | 0.0000 | 1.0000 | ← best |

> Early stopping triggered after 1 epoch(s).

---

## 4. Best Validation Performance (new model, threshold=0.5)

| Metric | Baseline | New model | Change |
|--------|--------:|--------:|--------|
| Accuracy | 0.8381 | 0.5 | ↓ 0.3381 |
| Recall | 0.9974 | 0.0 | ↓ 0.9974 |
| Specificity | 0.5726 | 1.0 | ↑ 0.4274 |
| F1 | 0.8851 | nan | → nan |
| Roc auc | 0.961 | 0.5 | ↓ 0.4610 |

> Note: Validation metrics are not test metrics. They guide model selection only.

---

## 5. Threshold Analysis on Validation Set

Thresholds explored on **validation data only** (test set not touched).

| Threshold | Accuracy | Precision | Recall | Specificity | F1 |
|----------:|---------:|----------:|-------:|------------:|---:|
| 0.10 | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.6667 ← recommended |
| 0.15 | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.6667 |
| 0.20 | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.6667 |
| 0.25 | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.6667 |
| 0.30 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.35 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.40 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.45 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.50 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.55 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.60 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.65 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.70 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.75 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.80 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.85 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.90 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.95 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

> Recommended threshold from validation analysis: **0.1**
> Selection criterion: highest accuracy while keeping recall ≥ 0.95.

---

## 6. Final Test Set Evaluation (one-time, untouched set)

### At default threshold (0.50)

| Metric | Baseline | New model | Change |
|--------|--------:|--------:|--------|
| Accuracy | 0.8381 | 0.4286 | ↓ 0.4095 |
| Recall | 0.9974 | 0.0 | ↓ 0.9974 |
| Specificity | 0.5726 | 1.0 | ↑ 0.4274 |
| F1 | 0.8851 | 0.0 | ↓ 0.8851 |
| Roc auc | 0.961 | 0.5 | ↓ 0.4610 |

Confusion matrix (default threshold):

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (3) | 3 TN | 0 FP |
| **Actual PNEUMONIA** (4) | 4 FN | 0 TP |

### At tuned threshold (0.1)

| Metric | Baseline | Tuned | Change |
|--------|--------:|------:|--------|
| Accuracy | 0.8381 | 0.5714 | ↓ 0.2667 |
| Recall | 0.9974 | 1.0 | ↑ 0.0026 |
| Specificity | 0.5726 | 0.0 | ↓ 0.5726 |
| F1 | 0.8851 | 0.7273 | ↓ 0.1578 |

Confusion matrix (tuned threshold):

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| **Actual NORMAL** (3) | 0 TN | 3 FP |
| **Actual PNEUMONIA** (4) | 0 FN | 4 TP |

---

## 7. Conclusions

### Did more training improve the model?

- ROC-AUC decreased slightly (0.961 → 0.5), suggesting possible overfitting.
- Specificity improved (0.5726 → 1.0): fewer normal images mis-classified as pneumonia.
- 90% accuracy target **not reached** at threshold 0.1 (accuracy = 0.5714). Further work (more data, architecture changes, or label revision) would be needed to consistently exceed 90%.

### Key distinction

ROC-AUC measures how well the model separates classes across all thresholds.
Recall and specificity depend on the chosen threshold.  A high ROC-AUC with
low specificity at 0.5 is expected when the model is optimised for recall.
Threshold tuning on validation data is the correct way to explore the tradeoff.

---

## 8. Limitations

- Only 3 more epochs trained (warm-start from epoch-3 checkpoint).
- Labels are lung-opacity annotations, not confirmed clinical diagnoses.
- Threshold was selected on validation data; generalisation to other
  populations is unknown.
- No Grad-CAM or explainability analysis yet.

---

_Generated by `src/finetune_pneumonia.py`_