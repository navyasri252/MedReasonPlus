"""
src/train_pneumonia_baseline.py
================================
Phase 6 — Pneumonia Image Classification Baseline for MedReason++

MODEL:   ResNet18 pretrained on ImageNet, fine-tuned for binary classification.
DATASET: Chest X-Ray Pneumonia (Kermany et al., Kaggle).

⚠ CLINICAL DISCLAIMER ⚠
This model is a research prototype only.  It has NOT been validated for
clinical use and must NOT be used to diagnose or treat patients.

USAGE
-----
    python src/train_pneumonia_baseline.py

    python src/train_pneumonia_baseline.py \\
        --dataset_root "archive (7)/chest_xray" \\
        --epochs 10 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --seed 42 \\
        --output_dir processed/phase6
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Optional sklearn / numpy (used for metrics only — not for training)
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Project module (importable from src/ or project root)
import sys
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import (
    IDX_TO_CLASS,
    build_datasets,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Return a ResNet18 with the final fully-connected layer replaced for
    binary classification.

    Parameters
    ----------
    pretrained  : bool  load ImageNet weights if True
    num_classes : int   output neurons (2 for NORMAL / PNEUMONIA)

    Returns
    -------
    nn.Module
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Run one full pass over the training set.

    Returns
    -------
    dict  {"loss": float, "accuracy": float}
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss":     total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total    if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluation loop (one epoch, no gradient)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on a DataLoader.

    Returns
    -------
    dict  {"loss": float, "accuracy": float,
            "all_labels": list[int], "all_probs": list[float]}
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_probs:  list[float] = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs  = model(images)
            loss     = criterion(outputs, labels)
            probs    = softmax(outputs)[:, 1]   # P(PNEUMONIA)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return {
        "loss":       total_loss / total if total > 0 else 0.0,
        "accuracy":   correct / total    if total > 0 else 0.0,
        "all_labels": all_labels,
        "all_probs":  all_probs,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: list[int],
    probs:  list[float],
    threshold: float = 0.5,
) -> dict:
    """
    Compute full classification metrics from ground-truth labels and
    predicted probabilities.

    Parameters
    ----------
    labels    : ground-truth integer labels (0 = NORMAL, 1 = PNEUMONIA)
    probs     : predicted probability for the PNEUMONIA class
    threshold : decision boundary (default 0.5)

    Returns
    -------
    dict with keys: accuracy, precision, recall, specificity, f1, roc_auc,
                    confusion_matrix, threshold, n_samples
    """
    if not _SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed — metrics not computed"}

    import numpy as np

    preds = [1 if p >= threshold else 0 for p in probs]
    y    = np.array(labels)
    yhat = np.array(preds)

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / len(y)
    precision   = precision_score(y, yhat, zero_division=0)
    recall      = recall_score(y, yhat, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y, yhat, zero_division=0)

    try:
        roc_auc = roc_auc_score(y, probs)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy":    round(float(accuracy),    4),
        "precision":   round(float(precision),   4),
        "recall":      round(float(recall),       4),
        "specificity": round(float(specificity), 4),
        "f1":          round(float(f1),           4),
        "roc_auc":     round(float(roc_auc),     4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "threshold":  threshold,
        "n_samples":  int(len(y)),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_history_csv(history: list[dict], path: Path) -> None:
    """Write epoch-by-epoch training history to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_metrics_json(metrics: dict, path: Path) -> None:
    """Write evaluation metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_training_report(
    config:       dict,
    split_counts: dict,
    history:      list[dict],
    test_metrics: dict,
    output_path:  Path,
) -> None:
    """
    Write a human-readable Markdown summary of training and evaluation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_epoch = max(history, key=lambda h: h.get("val_accuracy", 0), default={})
    cm         = test_metrics.get("confusion_matrix", {})

    lines = [
        "# Phase 6 — Pneumonia Classification Baseline",
        "",
        "> **⚠ CLINICAL DISCLAIMER:** This model is a research prototype.",
        "> It has NOT been validated for clinical use.",
        "",
        "---",
        "",
        "## 1. Configuration",
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
        "## 2. Dataset Split",
        "",
        "| Split | NORMAL | PNEUMONIA | Total |",
        "|-------|-------:|----------:|------:|",
    ]
    for split in ["train", "val", "test"]:
        sc = split_counts.get(split, {})
        lines.append(
            f"| {split} | {sc.get('NORMAL', 0):,} "
            f"| {sc.get('PNEUMONIA', 0):,} | {sc.get('total', 0):,} |"
        )
    lines += [
        "",
        "> Val set derived from training data via stratified split.",
        "> Test set is the original untouched folder.",
        "",
        "---",
        "",
        "## 3. Training History",
        "",
        "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |",
        "|------:|-----------:|----------:|---------:|--------:|",
    ]
    for row in history:
        lines.append(
            f"| {row['epoch']} "
            f"| {row['train_loss']:.4f} "
            f"| {row['train_accuracy']:.4f} "
            f"| {row['val_loss']:.4f} "
            f"| {row['val_accuracy']:.4f} |"
        )

    best_val_acc = best_epoch.get("val_accuracy", float("nan"))
    lines += [
        "",
        f"> Best validation accuracy: **{best_val_acc:.4f}** "
        f"(epoch {best_epoch.get('epoch', '?')})",
        "",
        "---",
        "",
        "## 4. Test Set Evaluation",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        val = test_metrics.get(metric, "N/A")
        lines.append(f"| {metric.replace('_', ' ').capitalize()} | {val} |")

    if cm:
        lines += [
            "",
            "### Confusion Matrix",
            "",
            "| | Predicted NORMAL | Predicted PNEUMONIA |",
            "|---|---------------:|--------------------:|",
            f"| **Actual NORMAL**    | {cm.get('tn', '?')} (TN) | {cm.get('fp', '?')} (FP) |",
            f"| **Actual PNEUMONIA** | {cm.get('fn', '?')} (FN) | {cm.get('tp', '?')} (TP) |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Limitations",
        "",
        "- Labels are radiologist-annotated lung opacity findings, not confirmed",
        "  clinical pneumonia diagnoses.",
        "- Bacterial vs viral pneumonia are not distinguished.",
        "- The validation set is very small in the original dataset (8 per class);",
        "  a stratified split was used instead.",
        "- No Grad-CAM or explainability analysis in this phase.",
        "- Model was trained on CPU — training may be slow on large epoch counts.",
        "",
        "---",
        "",
        "_Generated by `src/train_pneumonia_baseline.py`_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(
    dataset_root: Path,
    output_dir:   Path,
    epochs:       int   = 10,
    batch_size:   int   = 32,
    lr:           float = 1e-4,
    val_fraction: float = 0.15,
    seed:         int   = 42,
    pretrained:   bool  = True,
    num_workers:  int   = 0,
) -> dict:
    """
    Full training + evaluation pipeline.

    Parameters
    ----------
    dataset_root : Path to chest_xray/ root
    output_dir   : where to write model, history, metrics, and report
    epochs       : number of training epochs
    batch_size   : mini-batch size
    lr           : initial learning rate
    val_fraction : fraction of training data used for validation
    seed         : random seed (Python, numpy, torch)
    pretrained   : load ImageNet weights for ResNet18
    num_workers  : DataLoader workers (0 = main thread, safe on Windows)

    Returns
    -------
    dict  results including test_metrics, history, split_counts, model_path
    """
    # ---- Reproducibility ----
    torch.manual_seed(seed)
    if _SKLEARN_AVAILABLE:
        import numpy as np
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  Phase 6 — Pneumonia Baseline Training")
    print(f"  Device : {device}")
    print(f"  Epochs : {epochs}  Batch : {batch_size}  LR : {lr}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build datasets ----
    print("[1/5] Building datasets …")
    ds = build_datasets(dataset_root, val_fraction=val_fraction, seed=seed)
    sc = ds["split_counts"]
    print(f"       train : {sc['train']}")
    print(f"       val   : {sc['val']}")
    print(f"       test  : {sc['test']}")

    train_loader = DataLoader(
        ds["train_dataset"], batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        ds["val_dataset"], batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        ds["test_dataset"], batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    # ---- Build model ----
    print("[2/5] Building ResNet18 model …")
    model = build_model(pretrained=pretrained, num_classes=2)
    model = model.to(device)

    # ---- Loss and optimiser ----
    class_weights = ds["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ---- Training loop ----
    print("[3/5] Training …")
    history: list[dict] = []
    best_val_acc = -1.0
    best_model_path = output_dir / "best_model.pt"
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch":          epoch,
            "train_loss":     round(train_stats["loss"],     4),
            "train_accuracy": round(train_stats["accuracy"], 4),
            "val_loss":       round(val_stats["loss"],       4),
            "val_accuracy":   round(val_stats["accuracy"],   4),
            "epoch_time_s":   round(time.time() - t_epoch,  1),
        }
        history.append(row)

        improved = " ← best" if val_stats["accuracy"] > best_val_acc else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f}{improved}"
        )

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            torch.save(model.state_dict(), best_model_path)

    total_time = round(time.time() - t0, 1)
    print(f"\n  Training finished in {total_time}s.  Best val acc: {best_val_acc:.4f}")

    # ---- Load best model and evaluate on test set ----
    print("[4/5] Evaluating best model on test set …")
    model.load_state_dict(torch.load(best_model_path, map_location=device,
                                     weights_only=True))
    test_stats   = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_stats["all_labels"], test_stats["all_probs"])
    test_metrics["test_loss"]     = round(test_stats["loss"], 4)
    test_metrics["test_accuracy"] = round(test_stats["accuracy"], 4)

    print(f"  Accuracy : {test_metrics.get('accuracy', 'N/A')}")
    print(f"  Precision: {test_metrics.get('precision', 'N/A')}")
    print(f"  Recall   : {test_metrics.get('recall', 'N/A')}")
    print(f"  F1       : {test_metrics.get('f1', 'N/A')}")
    print(f"  ROC-AUC  : {test_metrics.get('roc_auc', 'N/A')}")

    # ---- Save outputs ----
    print("[5/5] Saving outputs …")
    history_path = output_dir / "training_history.csv"
    metrics_path = output_dir / "test_metrics.json"
    report_path  = Path("reports") / "phase6_baseline_report.md"

    save_history_csv(history, history_path)
    save_metrics_json(test_metrics, metrics_path)

    config = {
        "dataset_root": str(dataset_root),
        "model":        "ResNet18 (pretrained=True)",
        "epochs":       epochs,
        "batch_size":   batch_size,
        "lr":           lr,
        "val_fraction": val_fraction,
        "seed":         seed,
        "device":       str(device),
        "total_time_s": total_time,
    }
    write_training_report(config, sc, history, test_metrics, report_path)

    print(f"\n  Outputs saved to: {output_dir}")
    print(f"  Report saved to : {report_path}")
    print(f"\n{'='*60}\n")

    return {
        "history":      history,
        "test_metrics": test_metrics,
        "split_counts": sc,
        "model_path":   str(best_model_path),
        "config":       config,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Train a ResNet18 baseline on Chest X-Ray Pneumonia."
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_dir",   type=str, default="processed/phase6")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Skip ImageNet weights (faster but weaker)")
    args = parser.parse_args()

    if args.dataset_root:
        root = Path(args.dataset_root).resolve()
    else:
        # Auto-detect: search from project root
        project_root = Path(__file__).resolve().parent.parent
        root = None
        for candidate in project_root.rglob("train"):
            if (candidate / "NORMAL").is_dir() and (candidate / "PNEUMONIA").is_dir():
                root = candidate.parent
                break
        if root is None:
            raise FileNotFoundError(
                "Could not auto-detect dataset root. Pass --dataset_root explicitly."
            )

    run_training(
        dataset_root=root,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
