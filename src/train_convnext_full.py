"""
src/train_convnext_full.py
===========================
ConvNeXt-Tiny Fine-Tuning — MedReason++ (Local CPU-safe version)

Strategy for CPU / low-RAM training
-------------------------------------
ConvNeXt-Tiny has 27.8M parameters across 4 stages + classifier head.
On a 16 GB machine with only ~2 GB free, training ALL layers would exhaust
memory and take >2 hours per epoch.

Instead we use PARTIAL FINE-TUNING:
  - Stages 0–2 (early feature extractors): FROZEN
  - Stage 3 (deepest features) + classifier head: TRAINABLE
  - Trainable parameters: ~8.5M instead of 27.8M
  - Memory saving: ~65%   |   Speed improvement: ~3-4x

This is the standard approach used in medical imaging papers when
compute is limited.  The frozen layers already encode strong low-level
features from ImageNet; only the high-level discriminative features
are adapted to chest X-rays.

Usage
-----
    python src/train_convnext_full.py

    python src/train_convnext_full.py \\
        --dataset_root "archive (7)/chest_xray" \\
        --epochs 5 --batch_size 8 --lr 1e-4 --seed 42

⚠ CLINICAL DISCLAIMER: Research prototype. NOT for clinical use.
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
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

try:
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    _SK = True
except ImportError:
    _SK = False

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import (
    build_datasets,
    collect_image_paths,
    stratified_split,
)
from train_pneumonia_baseline import (
    compute_metrics,
    evaluate,
    save_history_csv,
    save_metrics_json,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Model: ConvNeXt-Tiny with partial fine-tuning
# ---------------------------------------------------------------------------

def build_convnext_partial(num_classes: int = 2) -> tuple[nn.Module, int]:
    """
    Load ConvNeXt-Tiny with ImageNet weights and freeze early stages.

    ConvNeXt-Tiny internal structure:
        features[0]  – stem (downsampling patch)
        features[1]  – stage 0  (3 blocks)   ← FROZEN
        features[2]  – downsampling
        features[3]  – stage 1  (3 blocks)   ← FROZEN
        features[4]  – downsampling
        features[5]  – stage 2  (9 blocks)   ← FROZEN
        features[6]  – downsampling
        features[7]  – stage 3  (3 blocks)   ← TRAINABLE
        classifier   – LayerNorm + Flatten + Linear(768→N) ← TRAINABLE

    Returns
    -------
    (model, n_trainable_params)
    """
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze stage 3 (features[7]) and classifier
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Replace final linear with 2-class head
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
    # New layer is trainable by default

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    return model, n_trainable, n_total


# ---------------------------------------------------------------------------
# Plots (saved as PNG if matplotlib available)
# ---------------------------------------------------------------------------

def save_plots(history: list[dict], test_labels, test_probs,
               output_dir: Path) -> None:
    """Save loss/accuracy curves, ROC curve, and confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc as sk_auc
        import numpy as np
    except ImportError:
        print("  matplotlib not installed — skipping plots.")
        return

    epochs = [r["epoch"] for r in history]

    # ── Loss curve ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train")
    axes[0].plot(epochs, [r["val_loss"]   for r in history], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, [r["train_accuracy"] for r in history], label="Train")
    axes[1].plot(epochs, [r["val_accuracy"]   for r in history], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "loss_accuracy_curves.png", dpi=120)
    plt.close(fig)

    # ── ROC curve ───────────────────────────────────────────────────────
    y = np.array(test_labels)
    fpr, tpr, _ = roc_curve(y, test_probs)
    roc_auc_val = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set"); ax.legend(); ax.grid(True)
    fig.savefig(output_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    # ── Confusion matrix ─────────────────────────────────────────────────
    yhat = np.array([1 if p >= 0.5 else 0 for p in test_probs])
    cm   = confusion_matrix(y, yhat, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["NORMAL","PNEUMONIA"])
    ax.set_yticklabels(["NORMAL","PNEUMONIA"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(
    dataset_root: Path,
    output_dir:   Path,
    epochs:       int   = 5,
    batch_size:   int   = 8,
    lr:           float = 1e-4,
    val_fraction: float = 0.15,
    seed:         int   = 42,
    num_workers:  int   = 0,
) -> dict:

    torch.manual_seed(seed)
    if _SK:
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  ConvNeXt-Tiny — MedReason++ Local Training")
    print(f"  Device    : {device}")
    print(f"  Batch     : {batch_size}  |  LR : {lr}  |  Epochs : {epochs}")
    print(f"  Strategy  : Partial fine-tune (stage-3 + head only)")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────
    print("[1/5] Building datasets …")
    ds = build_datasets(dataset_root, val_fraction=val_fraction, seed=seed)
    sc = ds["split_counts"]

    print(f"  Class mapping : NORMAL=0 (negative)  PNEUMONIA=1 (positive)")
    print(f"  Train         : NORMAL={sc['train']['NORMAL']}  "
          f"PNEUMONIA={sc['train']['PNEUMONIA']}  total={sc['train']['total']}")
    print(f"  Val (derived) : NORMAL={sc['val']['NORMAL']}  "
          f"PNEUMONIA={sc['val']['PNEUMONIA']}  total={sc['val']['total']}")
    print(f"  Test (locked) : NORMAL={sc['test']['NORMAL']}  "
          f"PNEUMONIA={sc['test']['PNEUMONIA']}  total={sc['test']['total']}")
    print()

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

    # Verify no data leakage
    train_paths = {str(r["path"]) for r in ds["train_records"]}
    val_paths   = {str(r["path"]) for r in ds["val_records"]}
    test_paths  = {str(r["path"]) for r in ds["test_records"]}
    assert train_paths.isdisjoint(val_paths),  "LEAK: train/val overlap"
    assert train_paths.isdisjoint(test_paths), "LEAK: train/test overlap"
    assert val_paths.isdisjoint(test_paths),   "LEAK: val/test overlap"
    print("  Data leakage check: PASSED (no overlap between splits)")

    # ── Model ─────────────────────────────────────────────────────────
    print("\n[2/5] Building ConvNeXt-Tiny (partial fine-tune) …")
    model, n_train, n_total = build_convnext_partial(num_classes=2)
    model = model.to(device)
    print(f"  Total params     : {n_total/1e6:.1f}M")
    print(f"  Trainable params : {n_train/1e6:.1f}M  "
          f"({100*n_train/n_total:.1f}% — stage-3 + head)")
    print(f"  Frozen params    : {(n_total-n_train)/1e6:.1f}M  "
          f"(stages 0–2, stem)")

    # ── Loss and optimiser ─────────────────────────────────────────────
    class_weights = ds["class_weights"].to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10,
    )

    # ── Training loop with early stopping on val ROC-AUC ──────────────
    print(f"\n[3/5] Training ({epochs} epochs, patience=2, early stop on val AUC) …\n")
    history:      list[dict] = []
    best_val_auc  = -1.0
    no_improve    = 0
    patience      = 2
    best_path     = output_dir / "best_model.pt"
    t0            = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_s = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_s   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if _SK and len(set(val_s["all_labels"])) > 1:
            val_auc = round(float(roc_auc_score(
                np.array(val_s["all_labels"]), val_s["all_probs"]
            )), 4)
        else:
            val_auc = round(val_s["accuracy"], 4)

        improved = val_auc > best_val_auc
        note     = "BEST" if improved else f"no improve {no_improve + 1}/{patience}"

        row = {
            "epoch":          epoch,
            "train_loss":     round(train_s["loss"], 4),
            "train_accuracy": round(train_s["accuracy"], 4),
            "val_loss":       round(val_s["loss"], 4),
            "val_accuracy":   round(val_s["accuracy"], 4),
            "val_roc_auc":    val_auc,
            "epoch_time_s":   round(time.time() - t_ep, 1),
        }
        history.append(row)

        print(
            f"  Ep {epoch:2d}/{epochs} | "
            f"train {row['train_loss']:.4f}/{row['train_accuracy']:.4f} | "
            f"val {row['val_loss']:.4f}/{row['val_accuracy']:.4f} "
            f"AUC {val_auc:.4f}  {note}"
        )

        if improved:
            best_val_auc = val_auc
            no_improve   = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping triggered (patience={patience}).")
                break

    total_time = round(time.time() - t0, 1)
    print(f"\n  Training done in {total_time}s  ({total_time/60:.1f} min)")
    print(f"  Best val AUC: {best_val_auc:.4f}")

    # ── Test evaluation (one time, untouched set) ─────────────────────
    print("\n[4/5] Evaluating best checkpoint on test set …")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    test_s   = evaluate(model, test_loader, criterion, device)
    metrics  = compute_metrics(test_s["all_labels"], test_s["all_probs"])
    metrics["test_loss"]      = round(test_s["loss"], 4)
    metrics["test_accuracy"]  = round(test_s["accuracy"], 4)
    metrics["epochs_trained"] = len(history)
    metrics["total_time_s"]   = total_time
    metrics["device"]         = str(device)
    metrics["trainable_params_M"] = round(n_train / 1e6, 1)

    cm = metrics["confusion_matrix"]
    print(f"\n  ── Test Results (threshold=0.50, n=624) ────────────")
    print(f"  Accuracy    : {metrics['accuracy']}")
    print(f"  Precision   : {metrics['precision']}")
    print(f"  Recall      : {metrics['recall']}")
    print(f"  Specificity : {metrics['specificity']}")
    print(f"  F1-score    : {metrics['f1']}")
    print(f"  ROC-AUC     : {metrics['roc_auc']}")
    print(f"  Confusion   : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  Total test  : {cm['tn']+cm['fp']+cm['fn']+cm['tp']}  (should be 624)")

    # ── Save outputs ──────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    save_history_csv(history, output_dir / "training_history.csv")
    save_metrics_json(metrics, output_dir / "test_metrics.json")

    # Save plots
    save_plots(history, test_s["all_labels"], test_s["all_probs"], output_dir)

    # Save config
    config = {
        "model":             "ConvNeXt-Tiny partial fine-tune (stage-3 + head)",
        "pretrained":        "ImageNet-1k",
        "trainable_params":  f"{n_train/1e6:.1f}M of {n_total/1e6:.1f}M",
        "dataset_root":      str(dataset_root),
        "val_fraction":      val_fraction,
        "seed":              seed,
        "epochs_max":        epochs,
        "epochs_run":        len(history),
        "batch_size":        batch_size,
        "lr":                lr,
        "device":            str(device),
        "total_time_s":      total_time,
        "class_mapping":     "NORMAL=0, PNEUMONIA=1",
        "test_set":          "original 624-image test folder, untouched",
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved: {output_dir}")
    print(f"{'='*60}\n")

    return {
        "history":      history,
        "test_metrics": metrics,
        "split_counts": sc,
        "model_path":   str(best_path),
        "config":       config,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ConvNeXt-Tiny partial fine-tune — MedReason++."
    )
    parser.add_argument("--dataset_root", default="archive (7)/chest_xray")
    parser.add_argument("--output_dir",   default="processed/convnext_v2")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    run_training(
        dataset_root=Path(args.dataset_root).resolve(),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
