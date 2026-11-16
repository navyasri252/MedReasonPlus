"""
src/evaluate_checkpoint.py
============================
Evaluate a saved checkpoint on the test set and write metrics + report.

Usage:
    python src/evaluate_checkpoint.py \\
        --model_type convnext \\
        --checkpoint processed/convnext/best_model.pt \\
        --dataset_root "archive (7)/chest_xray" \\
        --output_dir processed/convnext \\
        --history_csv processed/convnext/training_history.csv

    python src/evaluate_checkpoint.py \\
        --model_type cxformer \\
        --checkpoint processed/cxformer/best_model.pt ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chest_xray_dataset import build_datasets
from train_pneumonia_baseline import compute_metrics, evaluate, save_metrics_json, write_training_report


def evaluate_convnext(checkpoint: Path, dataset_root: Path, output_dir: Path,
                      history_csv: Path, val_fraction: float = 0.15, seed: int = 42,
                      batch_size: int = 32) -> dict:
    from train_convnext import build_convnext
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Evaluating ConvNeXt-Tiny checkpoint: {checkpoint}")

    ds = build_datasets(dataset_root, val_fraction=val_fraction, seed=seed)
    test_loader = DataLoader(ds["test_dataset"], batch_size=batch_size, shuffle=False)

    model = build_convnext(pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    criterion = nn.CrossEntropyLoss(weight=ds["class_weights"].to(device))
    test_s   = evaluate(model, test_loader, criterion, device)
    metrics  = compute_metrics(test_s["all_labels"], test_s["all_probs"])
    metrics["test_loss"]     = round(test_s["loss"], 4)
    metrics["test_accuracy"] = round(test_s["accuracy"], 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / "test_metrics.json")

    # Load history if available
    history = []
    if history_csv and Path(history_csv).exists():
        with open(history_csv) as f:
            history = list(csv.DictReader(f))
        # Convert numeric strings back to floats
        for row in history:
            for k in row:
                try:
                    row[k] = float(row[k])
                except ValueError:
                    pass

    config = {
        "model":        "ConvNeXt-Tiny (pretrained ImageNet-1k)",
        "checkpoint":   str(checkpoint),
        "dataset_root": str(dataset_root),
        "epochs_run":   len(history),
        "seed":         seed,
        "device":       str(device),
        "note":         "Evaluated from saved checkpoint (training stopped at epoch 4, val AUC=0.9992)",
    }
    write_training_report(
        config, ds["split_counts"], history, metrics,
        Path("reports") / "convnext_training_report.md",
    )
    return metrics


def evaluate_cxformer(checkpoint: Path, dataset_root: Path, output_dir: Path,
                      history_csv: Path, val_fraction: float = 0.15, seed: int = 42,
                      batch_size: int = 16) -> dict:
    from train_cxformer import build_cxformer, evaluate_cxf, CXformerDataset
    from chest_xray_dataset import collect_image_paths, stratified_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Evaluating CXformer-S checkpoint: {checkpoint}")

    model, processor = build_cxformer(freeze_backbone=True, num_classes=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)

    test_recs = collect_image_paths(dataset_root / "test")
    from chest_xray_dataset import compute_class_weights
    all_train = collect_image_paths(dataset_root / "train")
    train_recs, val_recs = stratified_split(all_train, val_fraction, seed)
    weights = compute_class_weights([r["label"] for r in train_recs])

    test_ds = CXformerDataset(test_recs, processor, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    test_s   = evaluate_cxf(model, test_loader, criterion, device)
    metrics  = compute_metrics(test_s["all_labels"], test_s["all_probs"])
    metrics["test_loss"]     = round(test_s["loss"], 4)
    metrics["test_accuracy"] = round(test_s["accuracy"], 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / "test_metrics.json")

    from collections import Counter
    def _c(recs):
        c = Counter(r["class_name"] for r in recs)
        return {"NORMAL": c.get("NORMAL",0), "PNEUMONIA": c.get("PNEUMONIA",0), "total": len(recs)}
    sc = {"train": _c(train_recs), "val": _c(val_recs), "test": _c(test_recs)}

    history = []
    if history_csv and Path(history_csv).exists():
        with open(history_csv) as f:
            history = list(csv.DictReader(f))
        for row in history:
            for k in row:
                try:
                    row[k] = float(row[k])
                except ValueError:
                    pass

    config = {
        "model":           "CXformer-S (m42-health/CXformer-small, CC BY-NC-4.0)",
        "checkpoint":      str(checkpoint),
        "backbone_frozen": "True",
        "epochs_run":      len(history),
        "seed":            seed,
        "device":          str(device),
    }
    write_training_report(
        config, sc, history, metrics,
        Path("reports") / "cxformer_training_report.md",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the test set.")
    parser.add_argument("--model_type",   required=True, choices=["convnext", "cxformer"])
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_root", default="archive (7)/chest_xray")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--history_csv",  default=None)
    parser.add_argument("--batch_size",   type=int, default=32)
    args = parser.parse_args()

    root    = Path(args.dataset_root).resolve()
    ckpt    = Path(args.checkpoint)
    out_dir = Path(args.output_dir)

    if args.model_type == "convnext":
        m = evaluate_convnext(ckpt, root, out_dir, args.history_csv, batch_size=args.batch_size)
    else:
        m = evaluate_cxformer(ckpt, root, out_dir, args.history_csv, batch_size=args.batch_size)

    print("\n  ── Test Results ──────────────────────")
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        print(f"  {k:12s}: {m.get(k, 'N/A')}")
    cm = m.get("confusion_matrix", {})
    if cm:
        print(f"  CM: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"\n  Metrics saved to: {out_dir}/test_metrics.json\n")


if __name__ == "__main__":
    main()
