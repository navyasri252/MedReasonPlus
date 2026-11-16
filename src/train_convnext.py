"""
src/train_convnext.py
======================
Phase 7a — ConvNeXt-Tiny Pneumonia Classifier for MedReason++

MODEL:   ConvNeXt-Tiny pretrained on ImageNet-1k (torchvision).
DATASET: Chest X-Ray Pneumonia (Kermany et al., Kaggle).
INPUT:   224×224 RGB images (same preprocessing as ResNet-18 baseline).

Uses the SAME dataset split, preprocessing, evaluation protocol, and
metric functions as the ResNet-18 baseline so results are directly
comparable.  The only difference from the baseline is the model architecture.

⚠ CLINICAL DISCLAIMER ⚠
Research prototype only.  NOT validated for clinical use.

USAGE
-----
    python src/train_convnext.py \\
        --dataset_root "archive (7)/chest_xray" \\
        --epochs 10 --patience 3 --lr 5e-5 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import build_datasets
from train_pneumonia_baseline import (
    compute_metrics,
    evaluate,
    save_history_csv,
    save_metrics_json,
    train_one_epoch,
    write_training_report,
)


# ---------------------------------------------------------------------------
# ConvNeXt-Tiny model factory
# ---------------------------------------------------------------------------

def build_convnext(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Return a ConvNeXt-Tiny with the final linear layer replaced for
    binary classification (NORMAL / PNEUMONIA).

    ConvNeXt-Tiny architecture:
      - Classifier head: LayerNorm → Flatten → Linear(768 → 1000)
      - We replace the final Linear with Linear(768 → num_classes)

    Parameters
    ----------
    pretrained  : bool  use ImageNet-1k weights
    num_classes : int   output neurons

    Returns
    -------
    nn.Module
    """
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    # classifier[-1] is the Linear(768, 1000) layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_convnext_training(
    dataset_root: Path,
    output_dir:   Path,
    epochs:       int   = 10,
    patience:     int   = 3,
    batch_size:   int   = 32,
    lr:           float = 5e-5,
    val_fraction: float = 0.15,
    seed:         int   = 42,
    pretrained:   bool  = True,
    num_workers:  int   = 0,
) -> dict:
    """
    Train and evaluate a ConvNeXt-Tiny model.

    Early stopping is based on validation ROC-AUC (patience epochs without
    improvement).  The test set is evaluated exactly once at the end.

    Parameters match train_pneumonia_baseline.run_training() for comparability.

    Returns
    -------
    dict  with history, test_metrics, split_counts, model_path, config
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  ConvNeXt-Tiny Training — MedReason++ Phase 7a")
    print(f"  Device : {device} | Epochs : {epochs} | Patience : {patience}")
    print(f"{'='*60}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets (same pipeline as ResNet-18) ----
    print("[1/6] Building datasets …")
    ds = build_datasets(dataset_root, val_fraction=val_fraction, seed=seed)
    sc = ds["split_counts"]
    print(f"       train={sc['train']['total']}  val={sc['val']['total']}  test={sc['test']['total']}")

    train_loader = DataLoader(ds["train_dataset"], batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(ds["val_dataset"],   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(ds["test_dataset"],  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    # ---- Model ----
    print("[2/6] Building ConvNeXt-Tiny …")
    model = build_convnext(pretrained=pretrained, num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       Parameters: {param_count:.1f}M")

    # ---- Loss, optimiser, scheduler ----
    class_weights = ds["class_weights"].to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # ---- Training loop with early stopping on val ROC-AUC ----
    print("[3/6] Training …")
    history:      list[dict] = []
    best_val_auc  = -1.0
    no_improve    = 0
    best_path     = output_dir / "best_model.pt"
    t0            = time.time()

    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_s = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_s   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Compute val ROC-AUC for early stopping
        if _has_sklearn and val_s["all_labels"] and len(set(val_s["all_labels"])) > 1:
            val_auc = round(float(roc_auc_score(
                np.array(val_s["all_labels"]), val_s["all_probs"]
            )), 4)
        else:
            val_auc = round(val_s["accuracy"], 4)

        improved = val_auc > best_val_auc
        note     = "← best" if improved else f"no improve {no_improve + 1}/{patience}"

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
            f"  Epoch {epoch:3d}/{epochs} | "
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
                print(f"\n  Early stopping (patience={patience}).")
                break

    total_time = round(time.time() - t0, 1)
    print(f"\n  Finished in {total_time}s.  Best val AUC: {best_val_auc:.4f}")

    # ---- Test evaluation ----
    print("[4/6] Evaluating on test set …")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_s  = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(test_s["all_labels"], test_s["all_probs"])
    metrics["test_loss"]     = round(test_s["loss"], 4)
    metrics["test_accuracy"] = round(test_s["accuracy"], 4)

    for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        print(f"  {m:12s}: {metrics.get(m, 'N/A')}")

    # ---- Save outputs ----
    print("[5/6] Saving outputs …")
    save_history_csv(history, output_dir / "training_history.csv")
    save_metrics_json(metrics,  output_dir / "test_metrics.json")

    config = {
        "model":        "ConvNeXt-Tiny (pretrained ImageNet-1k)",
        "dataset_root": str(dataset_root),
        "epochs_max":   epochs,
        "epochs_run":   len(history),
        "patience":     patience,
        "batch_size":   batch_size,
        "lr":           lr,
        "val_fraction": val_fraction,
        "seed":         seed,
        "device":       str(device),
        "total_time_s": total_time,
    }
    write_training_report(
        config, sc, history, metrics,
        Path("reports") / "convnext_training_report.md",
    )

    print("[6/6] Done.")
    print(f"\n  Outputs : {output_dir}")
    print(f"  Report  : reports/convnext_training_report.md\n")

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
        description="Phase 7a: Train ConvNeXt-Tiny on Chest X-Ray Pneumonia."
    )
    parser.add_argument("--dataset_root", default="archive (7)/chest_xray")
    parser.add_argument("--output_dir",   default="processed/convnext")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    run_convnext_training(
        dataset_root=Path(args.dataset_root).resolve(),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
