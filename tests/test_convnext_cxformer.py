"""
tests/test_convnext_cxformer.py
================================
Unit and integration tests for:
  - src/train_convnext.py   (ConvNeXt-Tiny)
  - src/train_cxformer.py   (CXformer-S)

Unit tests use a tiny synthetic dataset (no real data needed).
Integration tests auto-skip if the real dataset or checkpoints are absent.

Run with:
    pytest tests/test_convnext_cxformer.py -v
"""

import sys
import torch
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_convnext import build_convnext, run_convnext_training
from train_cxformer import (
    CXformerClassifier,
    CXformerDataset,
    build_cxformer,
    evaluate_cxf,
    run_cxformer_training,
    train_one_epoch_cxf,
)
from chest_xray_dataset import collect_image_paths, stratified_split

# Re-use the fake dataset builder from baseline tests
from test_pneumonia_baseline import _make_dataset


# ---------------------------------------------------------------------------
# ── ConvNeXt-Tiny tests ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestBuildConvNeXt:
    def test_output_classes(self):
        model = build_convnext(pretrained=False, num_classes=2)
        assert model.classifier[-1].out_features == 2

    def test_forward_shape(self):
        model = build_convnext(pretrained=False, num_classes=2)
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_no_pretrained(self):
        model = build_convnext(pretrained=False)
        assert model is not None

    def test_custom_num_classes(self):
        model = build_convnext(pretrained=False, num_classes=3)
        assert model.classifier[-1].out_features == 3


class TestConvNeXtEndToEnd:
    def test_runs_on_fake_dataset(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=8, n_pn=12)
        results = run_convnext_training(
            dataset_root=root,
            output_dir=tmp_path / "convnext_out",
            epochs=1,
            patience=1,
            batch_size=4,
            lr=1e-4,
            seed=0,
            pretrained=False,
        )
        assert "history" in results
        assert "test_metrics" in results
        assert len(results["history"]) >= 1

    def test_checkpoint_saved(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=8, n_pn=12)
        out = tmp_path / "convnext_ck"
        run_convnext_training(
            dataset_root=root, output_dir=out,
            epochs=1, patience=1, batch_size=4,
            lr=1e-4, seed=0, pretrained=False,
        )
        assert (out / "best_model.pt").exists()

    def test_output_files_created(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=8, n_pn=12)
        out = tmp_path / "convnext_files"
        run_convnext_training(
            dataset_root=root, output_dir=out,
            epochs=1, patience=1, batch_size=4,
            lr=1e-4, seed=0, pretrained=False,
        )
        assert (out / "training_history.csv").exists()
        assert (out / "test_metrics.json").exists()

    def test_test_split_count(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=8, n_pn=12)
        results = run_convnext_training(
            dataset_root=root, output_dir=tmp_path / "cx_counts",
            epochs=1, patience=1, batch_size=4,
            lr=1e-4, seed=0, pretrained=False,
        )
        # test folder has 3 NORMAL + 4 PNEUMONIA (from _make_dataset)
        assert results["split_counts"]["test"]["total"] == 7

    def test_no_train_val_overlap(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=30, n_pn=50)
        results = run_convnext_training(
            dataset_root=root, output_dir=tmp_path / "cx_overlap",
            epochs=1, patience=1, batch_size=8,
            lr=1e-4, seed=42, pretrained=False,
        )
        train_n = results["split_counts"]["train"]["total"]
        val_n   = results["split_counts"]["val"]["total"]
        # They come from the same pool (80 images) without overlap
        assert train_n + val_n == 80


# ---------------------------------------------------------------------------
# ── CXformer-S model tests ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _make_mock_backbone(hidden_size: int = 384):
    """
    Create a tiny mock backbone that mimics CXformer's HuggingFace API
    (returns an object with .last_hidden_state) so we can test the wrapper
    without downloading the real model.
    """
    class _MockConfig:
        hidden_size = 384

    class _MockOutput:
        def __init__(self, hs):
            self.last_hidden_state = hs

    class _MockBackbone(torch.nn.Module):
        config = _MockConfig()

        def forward(self, pixel_values, **kwargs):
            B = pixel_values.size(0)
            # Simulate [B, 1374, 384] output
            hs = torch.zeros(B, 1374, hidden_size)
            return _MockOutput(hs)

    return _MockBackbone()


class TestCXformerClassifier:
    def test_forward_shape_frozen(self):
        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=2,
                                   freeze_backbone=True, hidden_dim=384)
        model.eval()
        x = torch.zeros(2, 3, 518, 518)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_frozen_backbone_no_grad(self):
        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=2,
                                   freeze_backbone=True, hidden_dim=384)
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfrozen_backbone_has_grad(self):
        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=2,
                                   freeze_backbone=False, hidden_dim=384)
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_head_is_trainable(self):
        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=2,
                                   freeze_backbone=True, hidden_dim=384)
        for param in model.head.parameters():
            assert param.requires_grad

    def test_custom_num_classes(self):
        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=3,
                                   freeze_backbone=True, hidden_dim=384)
        x = torch.zeros(1, 3, 518, 518)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)


class TestCXformerDataset:
    """Test CXformerDataset with a mock processor."""

    def _mock_processor(self):
        """Return a processor-like callable that returns 518×518 tensors."""
        class _MockProcessor:
            def __call__(self, image, return_tensors="pt"):
                return {"pixel_values": torch.zeros(1, 3, 518, 518)}

        return _MockProcessor()

    def test_len(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=5, n_pn=5)
        records = collect_image_paths(root / "train")
        ds = CXformerDataset(records, self._mock_processor(), augment=False)
        assert len(ds) == 10

    def test_item_shape(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=5, n_pn=5)
        records = collect_image_paths(root / "train")
        ds = CXformerDataset(records, self._mock_processor(), augment=False)
        pv, label = ds[0]
        assert pv.shape == (3, 518, 518)
        assert label in (0, 1)

    def test_labels_property(self, tmp_path):
        root = _make_dataset(tmp_path, n_normal=5, n_pn=5)
        records = collect_image_paths(root / "train")
        ds = CXformerDataset(records, self._mock_processor())
        labels = ds.labels
        assert len(labels) == 10
        assert all(l in (0, 1) for l in labels)


class TestCXformerTrainEval:
    """Test train/eval loops with mock backbone and tiny dataset."""

    def _make_mock_model_and_loader(self, tmp_path, batch_size=2):
        root = _make_dataset(tmp_path, n_normal=4, n_pn=6)
        records = collect_image_paths(root / "train")

        class _MockProc:
            def __call__(self, image, return_tensors="pt"):
                return {"pixel_values": torch.zeros(1, 3, 518, 518)}

        ds = CXformerDataset(records, _MockProc(), augment=False)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=batch_size)

        backbone = _make_mock_backbone(384)
        model = CXformerClassifier(backbone, num_classes=2,
                                   freeze_backbone=True, hidden_dim=384)
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.head.parameters(), lr=1e-3)
        device = torch.device("cpu")
        return model, loader, criterion, optimizer, device

    def test_train_one_epoch_returns_loss_acc(self, tmp_path):
        model, loader, criterion, optimizer, device = \
            self._make_mock_model_and_loader(tmp_path)
        stats = train_one_epoch_cxf(model, loader, criterion, optimizer, device)
        assert "loss" in stats and "accuracy" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_evaluate_returns_labels_probs(self, tmp_path):
        model, loader, criterion, _, device = \
            self._make_mock_model_and_loader(tmp_path)
        stats = evaluate_cxf(model, loader, criterion, device)
        assert "all_labels" in stats and "all_probs" in stats
        assert len(stats["all_labels"]) == len(stats["all_probs"])

    def test_probs_in_valid_range(self, tmp_path):
        model, loader, criterion, _, device = \
            self._make_mock_model_and_loader(tmp_path)
        stats = evaluate_cxf(model, loader, criterion, device)
        for p in stats["all_probs"]:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Integration: real dataset, real model downloads (auto-skip if absent)
# ---------------------------------------------------------------------------

_REAL_CXR = PROJECT_ROOT / "archive (7)" / "chest_xray"
_REAL_PRESENT = _REAL_CXR.is_dir()


@pytest.mark.skipif(not _REAL_PRESENT, reason="Real CXR dataset not found.")
class TestRealConvNeXtIntegration:

    def test_convnext_1epoch_real(self, tmp_path):
        """1 epoch of ConvNeXt-Tiny on real data — verifies split counts and file creation."""
        results = run_convnext_training(
            dataset_root=_REAL_CXR,
            output_dir=tmp_path / "convnext_real",
            epochs=1,
            patience=1,
            batch_size=32,
            lr=5e-5,
            seed=42,
            pretrained=True,
        )
        assert results["split_counts"]["test"]["total"] == 624
        assert (tmp_path / "convnext_real" / "best_model.pt").exists()


@pytest.mark.skipif(not _REAL_PRESENT, reason="Real CXR dataset not found.")
class TestRealCXformerIntegration:

    def test_cxformer_loads_and_runs_1epoch(self, tmp_path):
        """
        Load CXformer-S from HuggingFace, run 1 epoch.
        Verifies: model loads, processor works at 518×518, split counts correct.
        """
        results = run_cxformer_training(
            dataset_root=_REAL_CXR,
            output_dir=tmp_path / "cxformer_real",
            epochs=1,
            patience=1,
            batch_size=8,
            lr=1e-4,
            seed=42,
            freeze_backbone=True,
        )
        assert results["split_counts"]["test"]["total"] == 624
        assert (tmp_path / "cxformer_real" / "best_model.pt").exists()
        # Verify test metrics contain all expected keys
        for key in ("accuracy", "recall", "specificity", "f1", "roc_auc"):
            assert key in results["test_metrics"]
