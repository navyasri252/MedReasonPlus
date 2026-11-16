"""
tests/test_pneumonia_baseline.py
==================================
Unit and integration tests for Phase 6:
  - src/chest_xray_dataset.py
  - src/train_pneumonia_baseline.py

All unit tests use a tiny synthetic fixture (a few 1-pixel JPEGs) so they
run fast with no access to the real dataset.

Integration tests run only when the real dataset is found.

Run with:
    pytest tests/test_pneumonia_baseline.py -v
"""

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chest_xray_dataset import (
    CLASS_TO_IDX,
    CLASSES,
    ChestXrayDataset,
    build_datasets,
    collect_image_paths,
    compute_class_weights,
    get_eval_transforms,
    get_train_transforms,
    stratified_split,
)
from train_pneumonia_baseline import (
    build_model,
    compute_metrics,
    evaluate,
    save_history_csv,
    save_metrics_json,
    train_one_epoch,
    write_training_report,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Minimal valid 1×1 white JPEG bytes
_TINY_JPEG = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
    b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
    b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e'
    b'af\xd8N\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
    b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
    b'\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03'
    b'\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12'
    b'!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1'
    b'\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTU'
    b'VWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93'
    b'\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9'
    b'\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6'
    b'\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2'
    b'\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7'
    b'\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a'
    b'(\x03\xff\xd9'
)


def _make_dataset(tmp_path: Path, n_normal: int = 10, n_pn: int = 20) -> Path:
    """
    Create a minimal fake Chest X-Ray Pneumonia dataset under tmp_path.

    Layout:
        tmp_path/chest_xray/
            train/NORMAL/     n_normal × *.jpeg
            train/PNEUMONIA/  n_pn     × *.jpeg
            val/NORMAL/       2 × *.jpeg
            val/PNEUMONIA/    2 × *.jpeg
            test/NORMAL/      3 × *.jpeg
            test/PNEUMONIA/   4 × *.jpeg
    """
    root = tmp_path / "chest_xray"

    _counter = [0]
    def _next():
        _counter[0] += 1
        return _counter[0]

    specs = {
        "train": {"NORMAL": n_normal, "PNEUMONIA": n_pn},
        "val":   {"NORMAL": 2,         "PNEUMONIA": 2},
        "test":  {"NORMAL": 3,         "PNEUMONIA": 4},
    }
    for split, classes in specs.items():
        for cls, count in classes.items():
            folder = root / split / cls
            folder.mkdir(parents=True)
            for _ in range(count):
                n = _next()
                (folder / f"img_{n:04d}.jpeg").write_bytes(_TINY_JPEG)

    return root


# ---------------------------------------------------------------------------
# Tests: chest_xray_dataset — collect_image_paths
# ---------------------------------------------------------------------------

class TestCollectImagePaths:
    def test_counts_correct(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        normal_count = sum(1 for r in records if r["class_name"] == "NORMAL")
        pn_count     = sum(1 for r in records if r["class_name"] == "PNEUMONIA")
        assert normal_count == 10
        assert pn_count     == 20

    def test_labels_are_integers(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        for rec in records:
            assert isinstance(rec["label"], int)
            assert rec["label"] in (0, 1)

    def test_class_to_idx_mapping(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        for rec in records:
            assert rec["label"] == CLASS_TO_IDX[rec["class_name"]]

    def test_paths_exist(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        for rec in records:
            assert rec["path"].exists()

    def test_missing_class_folder_raises(self, tmp_path):
        root = _make_dataset(tmp_path)
        import shutil
        shutil.rmtree(root / "train" / "PNEUMONIA")
        with pytest.raises(FileNotFoundError):
            collect_image_paths(root / "train")


# ---------------------------------------------------------------------------
# Tests: stratified_split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_correct_total_count(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        train, val = stratified_split(records, val_fraction=0.2, seed=42)
        assert len(train) + len(val) == len(records)

    def test_fraction_approximately_correct(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=100, n_pn=200)
        records = collect_image_paths(root / "train")
        _, val = stratified_split(records, val_fraction=0.15, seed=42)
        frac = len(val) / len(records)
        assert 0.10 <= frac <= 0.20

    def test_class_distribution_preserved(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=100, n_pn=100)
        records = collect_image_paths(root / "train")
        train, val = stratified_split(records, val_fraction=0.2, seed=42)
        # Both subsets should contain both classes
        train_labels = {r["label"] for r in train}
        val_labels   = {r["label"] for r in val}
        assert 0 in train_labels and 1 in train_labels
        assert 0 in val_labels   and 1 in val_labels

    def test_reproducibility(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=50, n_pn=50)
        records = collect_image_paths(root / "train")
        t1, v1 = stratified_split(records, val_fraction=0.2, seed=42)
        t2, v2 = stratified_split(records, val_fraction=0.2, seed=42)
        ids1 = sorted(str(r["path"]) for r in v1)
        ids2 = sorted(str(r["path"]) for r in v2)
        assert ids1 == ids2

    def test_invalid_fraction_raises(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        with pytest.raises(ValueError):
            stratified_split(records, val_fraction=1.5)
        with pytest.raises(ValueError):
            stratified_split(records, val_fraction=0.0)

    def test_no_overlap_between_train_and_val(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=50, n_pn=50)
        records = collect_image_paths(root / "train")
        train, val = stratified_split(records, val_fraction=0.2, seed=42)
        train_paths = {str(r["path"]) for r in train}
        val_paths   = {str(r["path"]) for r in val}
        assert train_paths.isdisjoint(val_paths)


# ---------------------------------------------------------------------------
# Tests: ChestXrayDataset
# ---------------------------------------------------------------------------

class TestChestXrayDataset:
    def test_len(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        ds = ChestXrayDataset(records, get_eval_transforms())
        assert len(ds) == 30  # 10 normal + 20 pn

    def test_item_shape(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        ds = ChestXrayDataset(records, get_eval_transforms(image_size=64))
        img, label = ds[0]
        assert img.shape == (3, 64, 64)  # RGB tensor
        assert isinstance(label, int)

    def test_labels_property(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        ds = ChestXrayDataset(records, get_eval_transforms())
        labels = ds.labels
        assert len(labels) == len(records)
        assert all(lbl in (0, 1) for lbl in labels)

    def test_image_converted_to_rgb(self, tmp_path):
        root = _make_dataset(tmp_path)
        records = collect_image_paths(root / "train")
        ds = ChestXrayDataset(records, get_eval_transforms(image_size=32))
        img, _ = ds[0]
        assert img.shape[0] == 3  # 3 channels = RGB


# ---------------------------------------------------------------------------
# Tests: compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_output_shape(self):
        labels = [0, 0, 0, 1, 1]
        w = compute_class_weights(labels)
        assert w.shape == (len(CLASSES),)

    def test_minority_class_gets_higher_weight(self):
        # 3 NORMAL (0), 7 PNEUMONIA (1) → NORMAL should have higher weight
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        w = compute_class_weights(labels)
        assert w[0] > w[1], f"Expected w[NORMAL]>w[PN] but got {w}"

    def test_balanced_classes_equal_weights(self):
        labels = [0, 0, 0, 1, 1, 1]
        w = compute_class_weights(labels)
        assert abs(w[0].item() - w[1].item()) < 1e-6

    def test_returns_float_tensor(self):
        labels = [0, 1, 0, 1]
        w = compute_class_weights(labels)
        assert w.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: build_datasets
# ---------------------------------------------------------------------------

class TestBuildDatasets:
    def test_returns_expected_keys(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = build_datasets(root, val_fraction=0.2, seed=42)
        for key in ("train_dataset", "val_dataset", "test_dataset",
                    "class_weights", "split_counts"):
            assert key in result

    def test_test_set_unchanged(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = build_datasets(root, val_fraction=0.2, seed=42)
        # Original test folder has 3 NORMAL + 4 PNEUMONIA = 7
        assert result["split_counts"]["test"]["total"] == 7

    def test_train_plus_val_equals_original_train(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = build_datasets(root, val_fraction=0.2, seed=42)
        sc = result["split_counts"]
        total = sc["train"]["total"] + sc["val"]["total"]
        assert total == 30  # 10 NORMAL + 20 PNEUMONIA from train folder

    def test_no_overlap_train_val(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = build_datasets(root, val_fraction=0.2, seed=42)
        train_paths = {str(r["path"]) for r in result["train_records"]}
        val_paths   = {str(r["path"]) for r in result["val_records"]}
        assert train_paths.isdisjoint(val_paths)

    def test_class_weights_length(self, tmp_path):
        root = _make_dataset(tmp_path)
        result = build_datasets(root, val_fraction=0.2, seed=42)
        assert len(result["class_weights"]) == 2


# ---------------------------------------------------------------------------
# Tests: build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_output_classes(self):
        model = build_model(pretrained=False, num_classes=2)
        assert model.fc.out_features == 2

    def test_forward_pass_shape(self):
        model = build_model(pretrained=False, num_classes=2)
        model.eval()
        dummy = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 2)

    def test_no_pretrained_works(self):
        model = build_model(pretrained=False)
        assert model is not None


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = [0, 0, 1, 1, 1]
        probs  = [0.1, 0.2, 0.9, 0.8, 0.7]
        m = compute_metrics(labels, probs)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0
        assert m["roc_auc"]   == 1.0

    def test_all_wrong_predictions(self):
        labels = [0, 0, 1, 1]
        probs  = [0.9, 0.9, 0.1, 0.1]
        m = compute_metrics(labels, probs)
        assert m["accuracy"] == 0.0

    def test_confusion_matrix_keys(self):
        labels = [0, 1, 0, 1]
        probs  = [0.1, 0.9, 0.2, 0.8]
        m = compute_metrics(labels, probs)
        assert set(m["confusion_matrix"].keys()) == {"tn", "fp", "fn", "tp"}

    def test_specificity_all_negatives_correct(self):
        labels = [0, 0, 1, 1]
        probs  = [0.1, 0.2, 0.9, 0.8]
        m = compute_metrics(labels, probs)
        assert m["specificity"] == 1.0

    def test_recall_equals_sensitivity(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs  = [0.1, 0.2, 0.1, 0.9, 0.8, 0.7]
        m = compute_metrics(labels, probs)
        assert m["recall"] == 1.0

    def test_n_samples(self):
        labels = [0, 1, 0, 1, 1]
        probs  = [0.2, 0.8, 0.3, 0.9, 0.7]
        m = compute_metrics(labels, probs)
        assert m["n_samples"] == 5

    def test_returns_error_dict_without_sklearn(self, monkeypatch):
        import train_pneumonia_baseline as mod
        monkeypatch.setattr(mod, "_SKLEARN_AVAILABLE", False)
        m = compute_metrics([0, 1], [0.1, 0.9])
        assert "error" in m


# ---------------------------------------------------------------------------
# Tests: train_one_epoch and evaluate (on tiny fake model)
# ---------------------------------------------------------------------------

class TestTrainingFunctions:
    """
    Run a single step through train_one_epoch and evaluate using a
    randomly-initialised ResNet18 (no pretrained weights) and a tiny dataset
    so tests stay fast.
    """

    @pytest.fixture()
    def tiny_setup(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=4, n_pn=6)
        from torch.utils.data import DataLoader
        result = build_datasets(root, val_fraction=0.3, seed=0)
        train_loader = DataLoader(result["train_dataset"], batch_size=4)
        val_loader   = DataLoader(result["val_dataset"],   batch_size=4)
        model = build_model(pretrained=False)
        device = torch.device("cpu")
        weights = result["class_weights"]
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, train_loader, val_loader, criterion, optimizer, device

    def test_train_one_epoch_returns_loss_and_acc(self, tmp_path, tiny_setup):
        model, train_loader, _, criterion, optimizer, device = tiny_setup
        stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_evaluate_returns_labels_and_probs(self, tmp_path, tiny_setup):
        model, _, val_loader, criterion, _, device = tiny_setup
        stats = evaluate(model, val_loader, criterion, device)
        assert "all_labels" in stats
        assert "all_probs"  in stats
        assert len(stats["all_labels"]) == len(stats["all_probs"])

    def test_evaluate_probs_in_0_1(self, tmp_path, tiny_setup):
        model, _, val_loader, criterion, _, device = tiny_setup
        stats = evaluate(model, val_loader, criterion, device)
        for p in stats["all_probs"]:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Tests: save helpers
# ---------------------------------------------------------------------------

class TestSaveHelpers:
    def test_history_csv_created(self, tmp_path):
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6,
             "train_accuracy": 0.7, "val_accuracy": 0.65},
        ]
        path = tmp_path / "history.csv"
        save_history_csv(history, path)
        assert path.exists()
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["epoch"] == "1"

    def test_metrics_json_created(self, tmp_path):
        metrics = {"accuracy": 0.92, "f1": 0.89}
        path = tmp_path / "metrics.json"
        save_metrics_json(metrics, path)
        assert path.exists()
        with path.open() as fh:
            data = json.load(fh)
        assert data["accuracy"] == 0.92

    def test_write_training_report_created(self, tmp_path):
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4,
             "train_accuracy": 0.7, "val_accuracy": 0.75},
        ]
        split_counts = {
            "train": {"NORMAL": 100, "PNEUMONIA": 200, "total": 300},
            "val":   {"NORMAL":  20, "PNEUMONIA":  40, "total":  60},
            "test":  {"NORMAL":  30, "PNEUMONIA":  50, "total":  80},
        }
        metrics = {
            "accuracy": 0.90, "precision": 0.88, "recall": 0.92,
            "specificity": 0.87, "f1": 0.90, "roc_auc": 0.95,
            "confusion_matrix": {"tn": 26, "fp": 4, "fn": 6, "tp": 44},
        }
        out = tmp_path / "report.md"
        write_training_report({"epochs": 1, "lr": 1e-4}, split_counts, history, metrics, out)
        assert out.exists()
        content = out.read_text()
        assert "Confusion Matrix" in content
        assert "Limitations" in content
        assert "CLINICAL DISCLAIMER" in content


# ---------------------------------------------------------------------------
# Integration tests on real dataset (auto-skipped if missing)
# ---------------------------------------------------------------------------

def _find_real_cxr() -> Path | None:
    candidates = [
        PROJECT_ROOT / "archive (7)" / "chest_xray",
        PROJECT_ROOT / "chest_xray",
    ]
    for c in candidates:
        if (c / "train" / "NORMAL").is_dir():
            return c
    return None


REAL_CXR = _find_real_cxr()


@pytest.mark.skipif(REAL_CXR is None, reason="Real CXR dataset not found.")
class TestRealDatasetIntegration:

    def test_collect_train_images(self):
        records = collect_image_paths(REAL_CXR / "train")
        n = sum(1 for r in records if r["class_name"] == "NORMAL")
        p = sum(1 for r in records if r["class_name"] == "PNEUMONIA")
        assert n == 1341
        assert p == 3875

    def test_stratified_split_counts(self):
        records = collect_image_paths(REAL_CXR / "train")
        train, val = stratified_split(records, val_fraction=0.15, seed=42)
        # Together they must equal total
        assert len(train) + len(val) == 5216
        # Val should be roughly 15%
        assert 700 <= len(val) <= 850

    def test_no_overlap_real(self):
        records = collect_image_paths(REAL_CXR / "train")
        train, val = stratified_split(records, val_fraction=0.15, seed=42)
        tp = {str(r["path"]) for r in train}
        vp = {str(r["path"]) for r in val}
        assert tp.isdisjoint(vp)

    def test_build_datasets_real(self):
        result = build_datasets(REAL_CXR, val_fraction=0.15, seed=42)
        sc = result["split_counts"]
        assert sc["test"]["total"] == 624
        assert sc["train"]["total"] + sc["val"]["total"] == 5216

    def test_class_weights_real(self):
        result = build_datasets(REAL_CXR, val_fraction=0.15, seed=42)
        w = result["class_weights"]
        # NORMAL (minority in train) should have higher weight than PNEUMONIA
        assert w[CLASS_TO_IDX["NORMAL"]] > w[CLASS_TO_IDX["PNEUMONIA"]]
