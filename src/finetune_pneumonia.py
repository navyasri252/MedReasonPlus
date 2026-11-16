"""
src/finetune_pneumonia.py
==========================
Phase 6b — Extended Fine-Tuning with Early Stopping and Threshold Analysis

PURPOSE
-------
Controlled follow-up to the 3-epoch baseline:
  - Warm-starts from the existing best_model.pt (epoch 3 checkpoint).
  - Trains for up to MAX_EPOCHS more epochs with early stopping on val ROC-AUC.
  - Tracks recall, specificity, ROC-AUC, accuracy on validation each epoch.
  - Performs threshold analysis on validation probabilities only.
  - Does NOT touch the test set until the very final evaluation.
  - Does NOT retrain repeatedly against test results.
  - Saves original Phase 6 outputs untouched.

OUTPUTS
-------
  processed/phase6b/
      best_model.pt               best checkpoint (by val ROC-AUC)
      training_history.csv        per-epoch metrics
      val_threshold_analysis.json sweep of thresholds on val set
      test_metrics.json           final test evaluation (one-time only)
  reports/
      phase6b_comparison_report.md

⚠ CLINICAL DISCLAIMER ⚠
This model is a research prototype. NOT validated for clinical use.

USAGE
-----
    python src/finetune_pneumonia.py \\
        --dataset_root "archive (7)/chest_xray" \\
        --checkpoint   processed/phase6/best_model.pt \\
        --max_epochs   10 \\
        --patience     3
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import build_datasets
from train_pneumonia_baseline import (
    build_model,
    compute_metrics,
    evaluate,
    save_history_csv,
    save_metrics_json,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Extended metrics helper (recall, specificity, ROC-AUC at a threshold)
# ---------------------------------------------------------------------------

def compute_val_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict:
    """Compact per-epoch validation metrics used for early-stopping decisions."""
    if not _SKLEARN_OK:
        return {"roc_auc": float("nan"), "recall": float("nan"),
                "specificity": float("nan"), "accuracy": float("nan")}
    import numpy as np
    y    = np.array(labels)
    yhat = np.array([1 if p >= threshold else 0 for p in probs])
    cm   = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc   = (tp + tn) / len(y)
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    return {
        "roc_auc":     round(float(auc),  4),
        "recall":      round(float(rec),  4),
        "specificity": round(float(spec), 4),
        "accuracy":    round(float(acc),  4),
    }


# ---------------------------------------------------------------------------
# Threshold sweep on validation set
# ---------------------------------------------------------------------------

def threshold_sweep(
    labels: list[int],
    probs:  list[float],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Evaluate recall, specificity, accuracy, F1, precision at many thresholds
    using only validation data.

    Parameters
    ----------
    labels     : ground-truth int labels
    probs      : predicted P(PNEUMONIA)
    thresholds : list of thresholds to test (default: 0.1 .. 0.9 in steps of 0.05)

    Returns
    -------
    list[dict]  one dict per threshold, sorted by threshold ascending
    """
    if thresholds is None:
        thresholds = [round(t / 100, 2) for t in range(10, 96, 5)]

    if not _SKLEARN_OK:
        return [{"threshold": t, "error": "sklearn missing"} for t in thresholds]

    import numpy as np
    y = np.array(labels)

    rows = []
    for t in thresholds:
        yhat = np.array([1 if p >= t else 0 for p in probs])
        cm   = confusion_matrix(y, yhat, labels=[0, 1])
        tn, fp, fn, tp_val = cm.ravel()
        acc  = (tp_val + tn) / len(y)
        rec  = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "threshold":   round(t, 2),
            "accuracy":    round(float(acc),  4),
            "precision":   round(float(prec), 4),
            "recall":      round(float(rec),  4),
            "specificity": round(float(spec), 4),
            "f1":          round(float(f1),   4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp_val),
        })
    return rows


# ---------------------------------------------------------------------------
# Markdown comparison report
# ---------------------------------------------------------------------------

def write_comparison_report(
    baseline_metrics:   dict,
    new_history:        list[dict],
    best_val_metrics:   dict,
    threshold_rows:     list[dict],
    recommended_thresh: float,
    test_at_default:    dict,
    test_at_tuned:      dict,
    split_counts:       dict,
    config:             dict,
    output_path:        Path,
) -> None:
    """Write a detailed Markdown comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 6b — Extended Fine-Tuning Comparison Report",
        "",
        "> **⚠ CLINICAL DISCLAIMER:** Research prototype. NOT validated for clinical use.",
        "",
        "---",
        "",
        "## 1. Experiment Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in config.items():
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Baseline Reference (Phase 6, 3 epochs, threshold=0.5)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        lines.append(f"| {m.replace('_',' ').capitalize()} | {baseline_metrics.get(m, 'N/A')} |")

    cm0 = baseline_metrics.get("confusion_matrix", {})
    if cm0:
        lines += [
            "",
            "Confusion matrix: "
            f"TN={cm0.get('tn')}, FP={cm0.get('fp')}, FN={cm0.get('fn')}, TP={cm0.get('tp')}",
        ]

    lines += [
        "",
        "---",
        "",
        "## 3. Extended Training History (warm-started from epoch-3 checkpoint)",
        "",
        "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val ROC-AUC | Val Recall | Val Spec | Note |",
        "|------:|-----------:|----------:|---------:|--------:|------------:|-----------:|---------:|------|",
    ]
    for row in new_history:
        note = row.get("note", "")
        lines.append(
            f"| {row['epoch']} "
            f"| {row['train_loss']:.4f} "
            f"| {row['train_accuracy']:.4f} "
            f"| {row.get('val_loss', 0):.4f} "
            f"| {row.get('val_accuracy', 0):.4f} "
            f"| {row.get('val_roc_auc', 0):.4f} "
            f"| {row.get('val_recall', 0):.4f} "
            f"| {row.get('val_specificity', 0):.4f} "
            f"| {note} |"
        )

    lines += [
        "",
        f"> Early stopping triggered after {len(new_history)} epoch(s).",
        "",
        "---",
        "",
        "## 4. Best Validation Performance (new model, threshold=0.5)",
        "",
        "| Metric | Baseline | New model | Change |",
        "|--------|--------:|--------:|--------|",
    ]
    for m in ["accuracy", "recall", "specificity", "f1", "roc_auc"]:
        old_v = baseline_metrics.get(m, float("nan"))
        new_v = best_val_metrics.get(m, float("nan"))
        try:
            delta = round(float(new_v) - float(old_v), 4)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | {arrow} {abs(delta):.4f} |")
        except Exception:
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | — |")

    lines += [
        "",
        "> Note: Validation metrics are not test metrics. They guide model selection only.",
        "",
        "---",
        "",
        "## 5. Threshold Analysis on Validation Set",
        "",
        "Thresholds explored on **validation data only** (test set not touched).",
        "",
        "| Threshold | Accuracy | Precision | Recall | Specificity | F1 |",
        "|----------:|---------:|----------:|-------:|------------:|---:|",
    ]
    for row in threshold_rows:
        marker = " ← recommended" if row["threshold"] == recommended_thresh else ""
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {row['accuracy']:.4f} "
            f"| {row['precision']:.4f} "
            f"| {row['recall']:.4f} "
            f"| {row['specificity']:.4f} "
            f"| {row['f1']:.4f}{marker} |"
        )

    lines += [
        "",
        f"> Recommended threshold from validation analysis: **{recommended_thresh}**",
        "> Selection criterion: highest accuracy while keeping recall ≥ 0.95.",
        "",
        "---",
        "",
        "## 6. Final Test Set Evaluation (one-time, untouched set)",
        "",
        "### At default threshold (0.50)",
        "",
        "| Metric | Baseline | New model | Change |",
        "|--------|--------:|--------:|--------|",
    ]
    for m in ["accuracy", "recall", "specificity", "f1", "roc_auc"]:
        old_v = baseline_metrics.get(m, float("nan"))
        new_v = test_at_default.get(m, float("nan"))
        try:
            delta = round(float(new_v) - float(old_v), 4)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | {arrow} {abs(delta):.4f} |")
        except Exception:
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | — |")

    cm_d = test_at_default.get("confusion_matrix", {})
    if cm_d:
        lines += [
            "",
            "Confusion matrix (default threshold):",
            "",
            "| | Predicted NORMAL | Predicted PNEUMONIA |",
            "|---|---:|---:|",
            f"| **Actual NORMAL** ({split_counts['test'].get('NORMAL', 234)}) "
            f"| {cm_d.get('tn')} TN | {cm_d.get('fp')} FP |",
            f"| **Actual PNEUMONIA** ({split_counts['test'].get('PNEUMONIA', 390)}) "
            f"| {cm_d.get('fn')} FN | {cm_d.get('tp')} TP |",
        ]

    lines += [
        "",
        f"### At tuned threshold ({recommended_thresh})",
        "",
        "| Metric | Baseline | Tuned | Change |",
        "|--------|--------:|------:|--------|",
    ]
    for m in ["accuracy", "recall", "specificity", "f1"]:
        old_v = baseline_metrics.get(m, float("nan"))
        new_v = test_at_tuned.get(m, float("nan"))
        try:
            delta = round(float(new_v) - float(old_v), 4)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | {arrow} {abs(delta):.4f} |")
        except Exception:
            lines.append(f"| {m.replace('_',' ').capitalize()} | {old_v} | {new_v} | — |")

    cm_t = test_at_tuned.get("confusion_matrix", {})
    if cm_t:
        lines += [
            "",
            "Confusion matrix (tuned threshold):",
            "",
            "| | Predicted NORMAL | Predicted PNEUMONIA |",
            "|---|---:|---:|",
            f"| **Actual NORMAL** ({split_counts['test'].get('NORMAL', 234)}) "
            f"| {cm_t.get('tn')} TN | {cm_t.get('fp')} FP |",
            f"| **Actual PNEUMONIA** ({split_counts['test'].get('PNEUMONIA', 390)}) "
            f"| {cm_t.get('fn')} FN | {cm_t.get('tp')} TP |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 7. Conclusions",
        "",
        "### Did more training improve the model?",
        "",
    ]

    new_auc = test_at_default.get("roc_auc", 0)
    old_auc = baseline_metrics.get("roc_auc", 0)
    new_spec = test_at_default.get("specificity", 0)
    old_spec = baseline_metrics.get("specificity", 0)
    new_acc = test_at_tuned.get("accuracy", 0)

    if float(new_auc) >= float(old_auc):
        lines.append(f"- ROC-AUC {'improved' if new_auc > old_auc else 'held steady'} "
                     f"({old_auc} → {new_auc}).")
    else:
        lines.append(f"- ROC-AUC decreased slightly ({old_auc} → {new_auc}), "
                     f"suggesting possible overfitting.")

    if float(new_spec) > float(old_spec):
        lines.append(f"- Specificity improved ({old_spec} → {new_spec}): "
                     f"fewer normal images mis-classified as pneumonia.")
    else:
        lines.append(f"- Specificity at 0.5 threshold did not improve "
                     f"({old_spec} → {new_spec}).")

    if float(new_acc) >= 0.90:
        lines.append(f"- **90% accuracy target achieved** on test set at threshold "
                     f"{recommended_thresh} (accuracy = {new_acc}).")
    else:
        lines.append(f"- 90% accuracy target **not reached** at threshold "
                     f"{recommended_thresh} (accuracy = {new_acc}). "
                     f"Further work (more data, architecture changes, or label revision) "
                     f"would be needed to consistently exceed 90%.")

    lines += [
        "",
        "### Key distinction",
        "",
        "ROC-AUC measures how well the model separates classes across all thresholds.",
        "Recall and specificity depend on the chosen threshold.  A high ROC-AUC with",
        "low specificity at 0.5 is expected when the model is optimised for recall.",
        "Threshold tuning on validation data is the correct way to explore the tradeoff.",
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "- Only 3 more epochs trained (warm-start from epoch-3 checkpoint).",
        "- Labels are lung-opacity annotations, not confirmed clinical diagnoses.",
        "- Threshold was selected on validation data; generalisation to other",
        "  populations is unknown.",
        "- No Grad-CAM or explainability analysis yet.",
        "",
        "---",
        "",
        "_Generated by `src/finetune_pneumonia.py`_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_finetuning(
    dataset_root:   Path,
    checkpoint:     Path,
    output_dir:     Path,
    baseline_json:  Path,
    max_epochs:     int   = 10,
    patience:       int   = 3,
    batch_size:     int   = 32,
    lr:             float = 5e-5,
    val_fraction:   float = 0.15,
    seed:           int   = 42,
    num_workers:    int   = 0,
) -> dict:
    """Run the extended fine-tuning experiment."""

    torch.manual_seed(seed)
    if _SKLEARN_OK:
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  Phase 6b — Extended Fine-Tuning (Early Stopping)")
    print(f"  Warm-start: {checkpoint}")
    print(f"  Device: {device}  |  Max epochs: {max_epochs}  |  Patience: {patience}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load baseline results ----
    with baseline_json.open() as fh:
        baseline_metrics = json.load(fh)
    print(f"[0/6] Baseline reference loaded: "
          f"AUC={baseline_metrics.get('roc_auc')}  "
          f"Recall={baseline_metrics.get('recall')}  "
          f"Specificity={baseline_metrics.get('specificity')}")

    # ---- Build datasets (same split, same seed) ----
    print("[1/6] Building datasets (same split as Phase 6) …")
    ds = build_datasets(dataset_root, val_fraction=val_fraction, seed=seed)
    sc = ds["split_counts"]
    print(f"       train={sc['train']['total']}  val={sc['val']['total']}  test={sc['test']['total']}")

    train_loader = DataLoader(ds["train_dataset"], batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(ds["val_dataset"],   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(ds["test_dataset"],  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    # ---- Load model from checkpoint ----
    print("[2/6] Loading model from checkpoint …")
    model = build_model(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    print("       Checkpoint loaded successfully.")

    # ---- Loss and optimiser (lower LR for fine-tuning) ----
    class_weights = ds["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine annealing so LR decays smoothly rather than a hard step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr / 10
    )

    # ---- Extended training loop ----
    print("[3/6] Fine-tuning with early stopping (metric: val ROC-AUC) …")
    history: list[dict] = []
    best_val_auc  = -1.0
    best_model_path = output_dir / "best_model.pt"
    no_improve = 0
    best_val_labels: list[int]  = []
    best_val_probs:  list[float] = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_ext = compute_val_metrics(val_stats["all_labels"], val_stats["all_probs"])
        improved = val_ext["roc_auc"] > best_val_auc
        note = ""

        if improved:
            best_val_auc   = val_ext["roc_auc"]
            no_improve     = 0
            best_val_labels = val_stats["all_labels"]
            best_val_probs  = val_stats["all_probs"]
            torch.save(model.state_dict(), best_model_path)
            note = "← best"
        else:
            no_improve += 1
            note = f"no improve {no_improve}/{patience}"

        row = {
            "epoch":            epoch,
            "train_loss":       round(train_stats["loss"],     4),
            "train_accuracy":   round(train_stats["accuracy"], 4),
            "val_loss":         round(val_stats["loss"],       4),
            "val_accuracy":     round(val_stats["accuracy"],   4),
            "val_roc_auc":      val_ext["roc_auc"],
            "val_recall":       val_ext["recall"],
            "val_specificity":  val_ext["specificity"],
            "epoch_time_s":     round(time.time() - t_ep, 1),
            "note":             note,
        }
        history.append(row)

        print(
            f"  Epoch {epoch:3d}/{max_epochs} | "
            f"train {row['train_loss']:.4f}/{row['train_accuracy']:.4f} | "
            f"val {row['val_loss']:.4f}/{row['val_accuracy']:.4f} | "
            f"AUC {row['val_roc_auc']:.4f} rec {row['val_recall']:.4f} "
            f"spec {row['val_specificity']:.4f}  {note}"
        )

        if no_improve >= patience:
            print(f"\n  Early stopping triggered (patience={patience}).")
            break

    total_time = round(time.time() - t0, 1)
    print(f"\n  Fine-tuning done in {total_time}s.  Best val AUC: {best_val_auc:.4f}")

    # ---- Threshold sweep on best val predictions ----
    print("[4/6] Running threshold analysis on validation set …")
    thresh_rows = threshold_sweep(best_val_labels, best_val_probs)

    # Recommended threshold: highest accuracy with recall >= 0.95
    qualifying = [r for r in thresh_rows if r["recall"] >= 0.95]
    if qualifying:
        recommended = max(qualifying, key=lambda r: r["accuracy"])["threshold"]
    else:
        # Fallback: highest F1
        recommended = max(thresh_rows, key=lambda r: r["f1"])["threshold"]

    print(f"  Recommended threshold (val, recall ≥ 0.95): {recommended}")

    # Best val metrics at default threshold
    best_val_metrics = compute_val_metrics(best_val_labels, best_val_probs, threshold=0.5)

    # ---- Final test evaluation ----
    print("[5/6] Final evaluation on test set (one-time) …")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_stats = evaluate(model, test_loader, criterion, device)

    test_at_default = compute_metrics(test_stats["all_labels"], test_stats["all_probs"], 0.5)
    test_at_tuned   = compute_metrics(test_stats["all_labels"], test_stats["all_probs"], recommended)

    print(f"  Default (0.50): acc={test_at_default['accuracy']} "
          f"rec={test_at_default['recall']} spec={test_at_default['specificity']} "
          f"AUC={test_at_default['roc_auc']}")
    print(f"  Tuned  ({recommended:.2f}): acc={test_at_tuned['accuracy']} "
          f"rec={test_at_tuned['recall']} spec={test_at_tuned['specificity']}")

    # ---- Save outputs ----
    print("[6/6] Saving outputs …")
    save_history_csv(history, output_dir / "training_history.csv")
    save_metrics_json(test_at_default, output_dir / "test_metrics_default_thresh.json")
    save_metrics_json(test_at_tuned,   output_dir / "test_metrics_tuned_thresh.json")

    with (output_dir / "val_threshold_analysis.json").open("w") as fh:
        json.dump({
            "recommended_threshold": recommended,
            "sweep": thresh_rows,
        }, fh, indent=2)

    config = {
        "warm_start_from": str(checkpoint),
        "dataset_root":    str(dataset_root),
        "model":           "ResNet18 (warm-start from epoch-3)",
        "max_epochs":      max_epochs,
        "patience":        patience,
        "lr":              lr,
        "batch_size":      batch_size,
        "seed":            seed,
        "device":          str(device),
        "total_time_s":    total_time,
        "epochs_trained":  len(history),
        "recommended_threshold": recommended,
    }

    write_comparison_report(
        baseline_metrics=baseline_metrics,
        new_history=history,
        best_val_metrics=best_val_metrics,
        threshold_rows=thresh_rows,
        recommended_thresh=recommended,
        test_at_default=test_at_default,
        test_at_tuned=test_at_tuned,
        split_counts=sc,
        config=config,
        output_path=Path("reports") / "phase6b_comparison_report.md",
    )

    print(f"  Saved to: {output_dir}")
    print(f"  Report  : reports/phase6b_comparison_report.md")
    print(f"\n{'='*60}\n")

    return {
        "history":            history,
        "best_val_metrics":   best_val_metrics,
        "test_at_default":    test_at_default,
        "test_at_tuned":      test_at_tuned,
        "threshold_rows":     thresh_rows,
        "recommended_thresh": recommended,
        "split_counts":       sc,
        "config":             config,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6b: Extended fine-tuning with early stopping."
    )
    parser.add_argument("--dataset_root", type=str,
                        default="archive (7)/chest_xray")
    parser.add_argument("--checkpoint",   type=str,
                        default="processed/phase6/best_model.pt")
    parser.add_argument("--baseline_json", type=str,
                        default="processed/phase6/test_metrics.json")
    parser.add_argument("--output_dir",   type=str,
                        default="processed/phase6b")
    parser.add_argument("--max_epochs",   type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    run_finetuning(
        dataset_root=Path(args.dataset_root).resolve(),
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        baseline_json=Path(args.baseline_json),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
