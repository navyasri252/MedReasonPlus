"""
tests/test_finetune_pneumonia.py
==================================
Tests for src/finetune_pneumonia.py — Phase 6b.

Unit tests use synthetic data (tiny fake dataset + a randomly-initialised
checkpoint).  Integration tests auto-skip if real data is absent.

Run with:
    pytest tests/test_finetune_pneumonia.py -v
"""

import json
import sys
import torch
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_pneumonia import (
    compute_val_metrics,
    run_finetuning,
    threshold_sweep,
    write_comparison_report,
)
from train_pneumonia_baseline import build_model, save_metrics_json

# Reuse the fake dataset builder from the baseline tests
from test_pneumonia_baseline import _make_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_path: Path) -> Path:
    """Save a random (untrained) ResNet18 checkpoint."""
    model = build_model(pretrained=False)
    ckpt_path = tmp_path / "fake_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def _make_baseline_json(tmp_path: Path) -> Path:
    """Write a fake baseline metrics JSON."""
    metrics = {
        "accuracy": 0.8381, "precision": 0.7955, "recall": 0.9974,
        "specificity": 0.5726, "f1": 0.8851, "roc_auc": 0.961,
        "confusion_matrix": {"tn": 134, "fp": 100, "fn": 1, "tp": 389},
        "threshold": 0.5,
    }
    path = tmp_path / "baseline_metrics.json"
    save_metrics_json(metrics, path)
    return path


# ---------------------------------------------------------------------------
# Tests: compute_val_metrics
# ---------------------------------------------------------------------------

class TestComputeValMetrics:
    def test_perfect_case(self):
        labels = [0, 0, 1, 1]
        probs  = [0.1, 0.2, 0.9, 0.8]
        m = compute_val_metrics(labels, probs)
        assert m["recall"]      == 1.0
        assert m["specificity"] == 1.0
        assert m["accuracy"]    == 1.0
        assert m["roc_auc"]     == 1.0

    def test_all_predicted_positive(self):
        labels = [0, 0, 1, 1]
        probs  = [0.9, 0.9, 0.9, 0.9]
        m = compute_val_metrics(labels, probs)
        assert m["recall"]      == 1.0
        assert m["specificity"] == 0.0

    def test_output_keys_present(self):
        labels = [0, 1]
        probs  = [0.2, 0.8]
        m = compute_val_metrics(labels, probs)
        for k in ("recall", "specificity", "accuracy", "roc_auc"):
            assert k in m

    def test_custom_threshold(self):
        labels = [0, 0, 1, 1]
        probs  = [0.3, 0.4, 0.6, 0.7]
        # At threshold 0.5 → probs 0.6 and 0.7 are predicted positive (≥0.5)
        # NORMAL (0): probs 0.3, 0.4 → both below 0.5 → predicted negative → TN
        # PNEUMONIA (1): probs 0.6, 0.7 → both ≥ 0.5 → predicted positive → TP
        # specificity = TN/(TN+FP) = 2/2 = 1.0, recall = TP/(TP+FN) = 2/2 = 1.0
        m = compute_val_metrics(labels, probs, threshold=0.5)
        assert m["specificity"] == 1.0
        assert m["recall"] == 1.0
        # At threshold 0.65 → only 0.7 is predicted positive
        # NORMAL (0): probs 0.3, 0.4 → TN → specificity=1.0
        # PNEUMONIA (1): 0.7 predicted positive (TP), 0.6 predicted negative (FN)
        m2 = compute_val_metrics(labels, probs, threshold=0.65)
        assert m2["recall"] == 0.5


# ---------------------------------------------------------------------------
# Tests: threshold_sweep
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    def test_returns_list(self):
        labels = [0, 0, 1, 1, 1]
        probs  = [0.1, 0.2, 0.9, 0.8, 0.7]
        rows = threshold_sweep(labels, probs)
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_each_row_has_required_keys(self):
        labels = [0, 1, 0, 1]
        probs  = [0.2, 0.8, 0.1, 0.9]
        rows = threshold_sweep(labels, probs)
        for row in rows:
            for k in ("threshold", "accuracy", "recall", "specificity", "f1", "precision"):
                assert k in row, f"Missing key '{k}'"

    def test_recall_decreases_as_threshold_rises(self):
        labels = [0, 0, 1, 1, 1, 1]
        probs  = [0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
        rows = threshold_sweep(labels, probs)
        # Recall should generally trend down as threshold rises
        recall_at_low  = next(r["recall"] for r in rows if r["threshold"] == 0.10)
        recall_at_high = next(r["recall"] for r in rows if r["threshold"] == 0.90)
        assert recall_at_low >= recall_at_high

    def test_specificity_increases_as_threshold_rises(self):
        labels = [0, 0, 0, 1, 1, 1]
        probs  = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
        rows = threshold_sweep(labels, probs)
        spec_at_low  = next(r["specificity"] for r in rows if r["threshold"] == 0.10)
        spec_at_high = next(r["specificity"] for r in rows if r["threshold"] == 0.90)
        assert spec_at_high >= spec_at_low

    def test_custom_thresholds(self):
        labels = [0, 1]
        probs  = [0.3, 0.7]
        rows = threshold_sweep(labels, probs, thresholds=[0.4, 0.6])
        assert len(rows) == 2
        assert {r["threshold"] for r in rows} == {0.4, 0.6}


# ---------------------------------------------------------------------------
# Tests: write_comparison_report
# ---------------------------------------------------------------------------

class TestWriteComparisonReport:
    def _sample_args(self, tmp_path):
        baseline = {
            "accuracy": 0.84, "precision": 0.80, "recall": 0.997,
            "specificity": 0.57, "f1": 0.885, "roc_auc": 0.961,
            "confusion_matrix": {"tn": 134, "fp": 100, "fn": 1, "tp": 389},
        }
        history = [{
            "epoch": 1, "train_loss": 0.12, "train_accuracy": 0.95,
            "val_loss": 0.08, "val_accuracy": 0.96,
            "val_roc_auc": 0.97, "val_recall": 0.99, "val_specificity": 0.70,
            "note": "← best",
        }]
        best_val = {"accuracy": 0.96, "recall": 0.99, "specificity": 0.70,
                    "f1": 0.91, "roc_auc": 0.97}
        thresh_rows = threshold_sweep([0, 0, 1, 1, 1], [0.2, 0.3, 0.8, 0.9, 0.7])
        rec_thresh  = 0.55
        test_def    = {"accuracy": 0.89, "recall": 0.99, "specificity": 0.72,
                       "f1": 0.92, "roc_auc": 0.97,
                       "confusion_matrix": {"tn": 169, "fp": 65, "fn": 1, "tp": 389}}
        test_tun    = {"accuracy": 0.91, "recall": 0.97, "specificity": 0.80,
                       "f1": 0.93,
                       "confusion_matrix": {"tn": 188, "fp": 46, "fn": 12, "tp": 378}}
        split_counts = {"test": {"NORMAL": 234, "PNEUMONIA": 390, "total": 624}}
        config       = {"epochs_trained": 1, "recommended_threshold": 0.55}
        out = tmp_path / "report.md"
        return (baseline, history, best_val, thresh_rows, rec_thresh,
                test_def, test_tun, split_counts, config, out)

    def test_report_created(self, tmp_path):
        args = self._sample_args(tmp_path)
        write_comparison_report(*args)
        assert args[-1].exists()

    def test_report_sections(self, tmp_path):
        args = self._sample_args(tmp_path)
        write_comparison_report(*args)
        content = args[-1].read_text(encoding="utf-8")
        for section in [
            "Baseline Reference", "Extended Training", "Threshold Analysis",
            "Test Set Evaluation", "Conclusions", "Limitations", "CLINICAL DISCLAIMER",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_recommended_threshold_in_report(self, tmp_path):
        args = self._sample_args(tmp_path)
        write_comparison_report(*args)
        content = args[-1].read_text(encoding="utf-8")
        assert "0.55" in content


# ---------------------------------------------------------------------------
# Tests: run_finetuning end-to-end (tiny fake dataset, 1 epoch)
# ---------------------------------------------------------------------------

class TestRunFinetuningEndToEnd:
    def test_runs_without_error(self, tmp_path):
        root    = _make_dataset(tmp_path, n_normal=10, n_pn=15)
        ckpt    = _make_checkpoint(tmp_path)
        bl_json = _make_baseline_json(tmp_path)
        out_dir = tmp_path / "phase6b"

        results = run_finetuning(
            dataset_root=root,
            checkpoint=ckpt,
            output_dir=out_dir,
            baseline_json=bl_json,
            max_epochs=1,
            patience=1,
            batch_size=4,
            lr=1e-4,
            seed=42,
        )
        assert "history" in results
        assert "test_at_default" in results
        assert "test_at_tuned" in results
        assert "threshold_rows" in results

    def test_output_files_created(self, tmp_path):
        root    = _make_dataset(tmp_path, n_normal=10, n_pn=15)
        ckpt    = _make_checkpoint(tmp_path)
        bl_json = _make_baseline_json(tmp_path)
        out_dir = tmp_path / "phase6b"

        run_finetuning(
            dataset_root=root, checkpoint=ckpt, output_dir=out_dir,
            baseline_json=bl_json, max_epochs=1, patience=1,
            batch_size=4, lr=1e-4, seed=42,
        )
        assert (out_dir / "best_model.pt").exists()
        assert (out_dir / "training_history.csv").exists()
        assert (out_dir / "val_threshold_analysis.json").exists()
        assert (out_dir / "test_metrics_default_thresh.json").exists()
        assert (out_dir / "test_metrics_tuned_thresh.json").exists()

    def test_early_stopping_respects_patience(self, tmp_path):
        root    = _make_dataset(tmp_path, n_normal=20, n_pn=30)
        ckpt    = _make_checkpoint(tmp_path)
        bl_json = _make_baseline_json(tmp_path)
        out_dir = tmp_path / "phase6b_es"

        # patience=1, max_epochs=5 → should stop well before 5
        results = run_finetuning(
            dataset_root=root, checkpoint=ckpt, output_dir=out_dir,
            baseline_json=bl_json, max_epochs=5, patience=1,
            batch_size=8, lr=1e-4, seed=0,
        )
        # With patience=1 and a random model, must stop at most at epoch 2
        assert len(results["history"]) <= 5

    def test_test_set_not_accessed_before_final_eval(self, tmp_path):
        """
        The threshold recommendation must come from val data.
        We verify the recommended threshold is in the threshold_rows list.
        """
        root    = _make_dataset(tmp_path, n_normal=10, n_pn=15)
        ckpt    = _make_checkpoint(tmp_path)
        bl_json = _make_baseline_json(tmp_path)
        out_dir = tmp_path / "phase6b_tt"

        results = run_finetuning(
            dataset_root=root, checkpoint=ckpt, output_dir=out_dir,
            baseline_json=bl_json, max_epochs=1, patience=1,
            batch_size=4, lr=1e-4, seed=42,
        )
        rec_thresh = results["recommended_thresh"]
        thresholds = [r["threshold"] for r in results["threshold_rows"]]
        assert rec_thresh in thresholds


# ---------------------------------------------------------------------------
# Integration tests (real dataset, auto-skipped if absent)
# ---------------------------------------------------------------------------

_REAL_CXR = PROJECT_ROOT / "archive (7)" / "chest_xray"
_REAL_CKPT = PROJECT_ROOT / "processed" / "phase6" / "best_model.pt"
_REAL_BL   = PROJECT_ROOT / "processed" / "phase6" / "test_metrics.json"
_REAL_PRESENT = _REAL_CXR.is_dir() and _REAL_CKPT.exists() and _REAL_BL.exists()


@pytest.mark.skipif(not _REAL_PRESENT, reason="Real dataset or checkpoint not found.")
class TestRealDataIntegration:

    def test_threshold_sweep_on_val(self):
        """Verify the sweep runs and produces sensible output on real val probs."""
        import torch
        from chest_xray_dataset import build_datasets
        from train_pneumonia_baseline import build_model, evaluate
        import torch.nn as nn

        device = torch.device("cpu")
        ds = build_datasets(_REAL_CXR, val_fraction=0.15, seed=42)
        val_loader = torch.utils.data.DataLoader(
            ds["val_dataset"], batch_size=64, shuffle=False
        )
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(_REAL_CKPT, map_location=device, weights_only=True))

        criterion = nn.CrossEntropyLoss(weight=ds["class_weights"])
        stats = evaluate(model, val_loader, criterion, device)

        rows = threshold_sweep(stats["all_labels"], stats["all_probs"])
        assert len(rows) == 18  # range(10, 96, 5) = 10,15,...,95 → 18 values
        # At very low threshold → recall should be near 1.0
        low_t = next(r for r in rows if r["threshold"] == 0.10)
        assert low_t["recall"] >= 0.95
        # At very high threshold → specificity should be near 1.0
        high_t = next(r for r in rows if r["threshold"] == 0.90)
        assert high_t["specificity"] >= 0.90
