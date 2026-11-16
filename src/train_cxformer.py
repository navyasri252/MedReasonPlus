"""
src/train_cxformer.py
======================
Phase 7b — CXformer-S Pneumonia Classifier for MedReason++

MODEL:   CXformer-S (m42-health/CXformer-small on HuggingFace).
         22M parameter ViT-S adapted from DINOv2 and pretrained on 600K+
         chest X-rays from 5 public datasets.
DATASET: Chest X-Ray Pneumonia (Kermany et al., Kaggle).

⚠ IMPORTANT DIFFERENCES from ResNet-18 / ConvNeXt-Tiny ⚠
  - Input size: 518×518 (CXformer's own AutoImageProcessor, not 224×224).
  - CXformer produces a sequence of patch tokens; we use the CLS token
    (index 0 of last_hidden_state) as the image feature vector.
  - The backbone is FROZEN during training; only the classification head
    is trained.  This is consistent with how CXformer is used in the
    original paper (linear probing / fine-tuning head on top).
  - Full fine-tuning is also supported via --unfreeze_backbone.

⚠ LICENSE ⚠
  CXformer is released under CC BY-NC-4.0.  It is for RESEARCH ONLY
  and must NOT be used in clinical or commercial settings.

⚠ CLINICAL DISCLAIMER ⚠
  Research prototype only.  NOT validated for clinical use.

DEPENDENCY
----------
  pip install transformers opencv-python-headless

USAGE
-----
    python src/train_cxformer.py \\
        --dataset_root "archive (7)/chest_xray" \\
        --epochs 10 --patience 3 --lr 1e-4 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import (
    CLASS_TO_IDX,
    CLASSES,
    collect_image_paths,
    stratified_split,
    compute_class_weights,
)
from train_pneumonia_baseline import (
    compute_metrics,
    save_history_csv,
    save_metrics_json,
    write_training_report,
)

# ---------------------------------------------------------------------------
# CXformer-specific dataset
# ---------------------------------------------------------------------------

class CXformerDataset(Dataset):
    """
    Dataset that applies CXformer's own AutoImageProcessor.

    CXformer uses 518×518 inputs and expects the processor's normalisation.
    We cannot reuse ChestXrayDataset (which applies 224×224 + ImageNet stats).

    Parameters
    ----------
    records   : list[dict]  from collect_image_paths()
    processor : AutoImageProcessor
    augment   : bool  apply training-time augmentation before processor
    """

    def __init__(self, records: list[dict], processor, augment: bool = False):
        self.records   = records
        self.processor = processor
        self.augment   = augment
        self._aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self._aug is not None:
            img = self._aug(img)
        inputs = self.processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 518, 518]
        return pixel_values, rec["label"]

    @property
    def labels(self) -> list[int]:
        return [r["label"] for r in self.records]


# ---------------------------------------------------------------------------
# CXformer wrapper model (backbone + linear head)
# ---------------------------------------------------------------------------

class CXformerClassifier(nn.Module):
    """
    CXformer-S backbone with a binary classification head.

    The backbone produces a sequence of patch tokens.  The CLS token
    (position 0 of last_hidden_state) is extracted and passed through a
    linear classification head.

    Parameters
    ----------
    backbone          : the HuggingFace AutoModel (CXformer-S)
    num_classes       : int (default 2)
    freeze_backbone   : bool  freeze backbone weights (linear probing mode)
    hidden_dim        : int   feature dimension of CLS token (384 for CXformer-S)
    """

    def __init__(
        self,
        backbone,
        num_classes: int  = 2,
        freeze_backbone: bool = True,
        hidden_dim: int   = 384,
    ):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Linear(hidden_dim, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]  # [B, 384]
        return self.head(cls_token)                  # [B, num_classes]


# ---------------------------------------------------------------------------
# Build CXformer model
# ---------------------------------------------------------------------------

def build_cxformer(freeze_backbone: bool = True, num_classes: int = 2):
    """
    Load CXformer-S from HuggingFace and wrap it with a classification head.

    Parameters
    ----------
    freeze_backbone : bool  freeze backbone (linear probing); set False for full fine-tune
    num_classes     : int

    Returns
    -------
    (CXformerClassifier, AutoImageProcessor)

    Raises
    ------
    ImportError  if `transformers` is not installed
    RuntimeError if the HuggingFace model cannot be loaded
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError(
            "The `transformers` package is required to load CXformer-S.\n"
            "Install it with:  pip install transformers"
        ) from exc

    model_id = "m42-health/CXformer-small"
    print(f"       Loading CXformer-S from HuggingFace: {model_id}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        backbone  = AutoModel.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CXformer-S from '{model_id}'.\n"
            f"Error: {exc}\n"
            "Check your internet connection and that `transformers` and "
            "`opencv-python-headless` are installed."
        ) from exc

    # CXformer-S hidden dim is 384 (ViT-S base)
    hidden_dim = backbone.config.hidden_size
    model = CXformerClassifier(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
    return model, processor


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch_cxf(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Training loop identical in structure to train_pneumonia_baseline."""
    model.train()
    total_loss = correct = total = 0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)
        preds  = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += pixel_values.size(0)

    return {
        "loss":     total_loss / total if total > 0 else 0.0,
        "accuracy": correct    / total if total > 0 else 0.0,
    }


def evaluate_cxf(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluation loop that collects labels and P(PNEUMONIA) probabilities."""
    model.eval()
    total_loss = correct = total = 0
    all_labels: list[int]  = []
    all_probs:  list[float] = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)

            outputs = model(pixel_values)
            loss    = criterion(outputs, labels)
            probs   = softmax(outputs)[:, 1]

            total_loss += loss.item() * pixel_values.size(0)
            preds  = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += pixel_values.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return {
        "loss":       total_loss / total if total > 0 else 0.0,
        "accuracy":   correct    / total if total > 0 else 0.0,
        "all_labels": all_labels,
        "all_probs":  all_probs,
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_cxformer_training(
    dataset_root:     Path,
    output_dir:       Path,
    epochs:           int   = 10,
    patience:         int   = 3,
    batch_size:       int   = 16,   # smaller default: 518×518 images are much larger
    lr:               float = 1e-4,
    val_fraction:     float = 0.15,
    seed:             int   = 42,
    freeze_backbone:  bool  = True,
    num_workers:      int   = 0,
) -> dict:
    """
    Train and evaluate a CXformer-S based classifier.

    By default the backbone is frozen (linear probing) — only the head
    is updated.  This is appropriate on CPU / limited compute because
    CXformer's backbone has already been pretrained on CXR data.

    Returns
    -------
    dict  with history, test_metrics, split_counts, model_path, config
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  CXformer-S Training — MedReason++ Phase 7b")
    print(f"  Device: {device} | Epochs: {epochs} | Patience: {patience}")
    print(f"  Backbone frozen: {freeze_backbone} (linear probing)")
    print(f"  Input: 518×518 (CXformer processor) — different from 224×224 baseline")
    print(f"  License: CC BY-NC-4.0 — research use only")
    print(f"{'='*60}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model and processor ----
    print("[1/6] Loading CXformer-S …")
    model, processor = build_cxformer(freeze_backbone=freeze_backbone, num_classes=2)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_p   = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"       Parameters: {total_p:.1f}M total, {trainable:.1f}M trainable")

    # ---- Datasets (CXformer-specific processor) ----
    print("[2/6] Building datasets with CXformer processor (518×518) …")
    train_folder = dataset_root / "train"
    test_folder  = dataset_root / "test"

    all_train = collect_image_paths(train_folder)
    train_recs, val_recs = stratified_split(all_train, val_fraction, seed)
    test_recs  = collect_image_paths(test_folder)

    train_ds = CXformerDataset(train_recs, processor, augment=True)
    val_ds   = CXformerDataset(val_recs,   processor, augment=False)
    test_ds  = CXformerDataset(test_recs,  processor, augment=False)

    from collections import Counter
    def _counts(recs):
        c = Counter(r["class_name"] for r in recs)
        return {"NORMAL": c.get("NORMAL", 0), "PNEUMONIA": c.get("PNEUMONIA", 0),
                "total": len(recs)}

    sc = {"train": _counts(train_recs), "val": _counts(val_recs), "test": _counts(test_recs)}
    print(f"       train={sc['train']['total']}  val={sc['val']['total']}  test={sc['test']['total']}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---- Loss, optimiser ----
    class_weights = compute_class_weights(train_ds.labels).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    # Only optimise trainable parameters (head if backbone is frozen)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    # ---- Training loop ----
    print("[3/6] Training …")
    history:     list[dict] = []
    best_val_auc = -1.0
    no_improve   = 0
    best_path    = output_dir / "best_model.pt"
    t0           = time.time()

    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    for epoch in range(1, epochs + 1):
        t_ep    = time.time()
        train_s = train_one_epoch_cxf(model, train_loader, criterion, optimizer, device)
        val_s   = evaluate_cxf(model, val_loader, criterion, device)
        scheduler.step()

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
    test_s  = evaluate_cxf(model, test_loader, criterion, device)
    metrics = compute_metrics(test_s["all_labels"], test_s["all_probs"])
    metrics["test_loss"]     = round(test_s["loss"], 4)
    metrics["test_accuracy"] = round(test_s["accuracy"], 4)

    for m in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        print(f"  {m:12s}: {metrics.get(m, 'N/A')}")

    # ---- Save ----
    print("[5/6] Saving outputs …")
    save_history_csv(history, output_dir / "training_history.csv")
    save_metrics_json(metrics,  output_dir / "test_metrics.json")

    config = {
        "model":             "CXformer-S (m42-health/CXformer-small, CC BY-NC-4.0)",
        "input_size":        "518×518 (CXformer AutoImageProcessor)",
        "backbone_frozen":   str(freeze_backbone),
        "dataset_root":      str(dataset_root),
        "epochs_max":        epochs,
        "epochs_run":        len(history),
        "patience":          patience,
        "batch_size":        batch_size,
        "lr":                lr,
        "val_fraction":      val_fraction,
        "seed":              seed,
        "device":            str(device),
        "total_time_s":      total_time,
        "license_note":      "CC BY-NC-4.0 — research use only",
    }
    write_training_report(
        config, sc, history, metrics,
        Path("reports") / "cxformer_training_report.md",
    )

    print("[6/6] Done.")
    print(f"\n  Outputs : {output_dir}")
    print(f"  Report  : reports/cxformer_training_report.md\n")

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
        description="Phase 7b: Train CXformer-S on Chest X-Ray Pneumonia."
    )
    parser.add_argument("--dataset_root",     default="archive (7)/chest_xray")
    parser.add_argument("--output_dir",       default="processed/cxformer")
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--patience",         type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--val_fraction",     type=float, default=0.15)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--unfreeze_backbone", action="store_true",
                        help="Full fine-tuning (slow on CPU, use with GPU)")
    args = parser.parse_args()

    run_cxformer_training(
        dataset_root=Path(args.dataset_root).resolve(),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        freeze_backbone=not args.unfreeze_backbone,
    )


if __name__ == "__main__":
    main()
