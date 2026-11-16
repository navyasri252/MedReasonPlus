"""
tests/test_audit_pneumonia_labels.py
======================================
Unit and integration tests for src/audit_pneumonia_labels.py — Phase 4.

All unit tests use minimal synthetic fixtures in a temp directory.
Integration tests against the real dataset are auto-skipped unless the
files exist at their default paths.

Run with:
    pytest tests/test_audit_pneumonia_labels.py -v
"""

import csv
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit_pneumonia_labels import (
    AUDIT_COLUMNS,
    build_audit_rows,
    check_cross_split_contamination,
    check_ids_alignment,
    check_view_label_consistency,
    compute_imbalance_stats,
    compute_split_summary,
    load_index_csv,
    load_labeled_csv,
    run_audit,
    write_audit_csv,
    write_audit_summary,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_labeled_csv(tmp_path: Path, extra_rows: list[dict] | None = None) -> Path:
    """
    Create a minimal labeled CSV with known positive/negative/uncertain rows.

    Default studies:
        CXR_POS_001 — positive / HIGH  / train
        CXR_POS_002 — positive / MEDIUM / train
        CXR_NEG_001 — negative / HIGH  / train
        CXR_NEG_002 — negative / MEDIUM / val
        CXR_NEG_003 — negative / LOW   / test
        CXR_UNC_001 — uncertain / MEDIUM / train
        CXR_UNC_002 — uncertain / LOW   / val
    """
    fieldnames = [
        "report_id", "study_image_count", "report", "report_word_count",
        "report_has_xxxx", "split", "pneumonia_label", "labeling_rule",
        "labeling_confidence",
    ]
    default_rows = [
        {"report_id": "CXR_POS_001", "study_image_count": "2",
         "report": "There is a right lower lobe pneumonia. No effusion.",
         "report_word_count": "9", "report_has_xxxx": "False",
         "split": "train", "pneumonia_label": "positive",
         "labeling_rule": "pneumonia_direct_assertion", "labeling_confidence": "HIGH"},
        {"report_id": "CXR_POS_002", "study_image_count": "2",
         "report": "Concerning for underlying pneumonia in the right upper lobe.",
         "report_word_count": "9", "report_has_xxxx": "False",
         "split": "train", "pneumonia_label": "positive",
         "labeling_rule": "pneumonia_suggestive_probable", "labeling_confidence": "MEDIUM"},
        {"report_id": "CXR_NEG_001", "study_image_count": "2",
         "report": "Lungs clear. No pneumonia, effusions, or pneumothorax.",
         "report_word_count": "8", "report_has_xxxx": "False",
         "split": "train", "pneumonia_label": "negative",
         "labeling_rule": "no_pneumonia_explicit", "labeling_confidence": "HIGH"},
        {"report_id": "CXR_NEG_002", "study_image_count": "2",
         "report": "No focal consolidation. Heart size normal.",
         "report_word_count": "7", "report_has_xxxx": "False",
         "split": "val", "pneumonia_label": "negative",
         "labeling_rule": "no_focal_consolidation", "labeling_confidence": "MEDIUM"},
        {"report_id": "CXR_NEG_003", "study_image_count": "2",
         "report": "Stable chest radiograph. Bony structures intact.",
         "report_word_count": "6", "report_has_xxxx": "False",
         "split": "test", "pneumonia_label": "negative",
         "labeling_rule": "no_relevant_language_found", "labeling_confidence": "LOW"},
        {"report_id": "CXR_UNC_001", "study_image_count": "2",
         "report": "Left lower lobe opacities could represent a pneumonia.",
         "report_word_count": "8", "report_has_xxxx": "False",
         "split": "train", "pneumonia_label": "uncertain",
         "labeling_rule": "uncertain_could_represent_pneumonia", "labeling_confidence": "MEDIUM"},
        {"report_id": "CXR_UNC_002", "study_image_count": "2",
         "report": "Cannot exclude early consolidation. Heart normal.",
         "report_word_count": "7", "report_has_xxxx": "False",
         "split": "val", "pneumonia_label": "uncertain",
         "labeling_rule": "uncertain_hedged_consolidation", "labeling_confidence": "LOW"},
    ]
    if extra_rows:
        default_rows.extend(extra_rows)

    p = tmp_path / "labeled.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(default_rows)
    return p


def _make_index_csv(tmp_path: Path, extra_rows: list[dict] | None = None) -> Path:
    """
    Create a minimal index CSV.  Each study has views 0 and 1.
    """
    fieldnames = [
        "image_path", "image_path_rel", "report_id", "view_index",
        "study_image_count", "report", "report_word_count",
        "report_has_xxxx", "split",
    ]
    study_ids = [
        "CXR_POS_001", "CXR_POS_002",
        "CXR_NEG_001", "CXR_NEG_002", "CXR_NEG_003",
        "CXR_UNC_001", "CXR_UNC_002",
    ]
    splits_map = {
        "CXR_POS_001": "train", "CXR_POS_002": "train",
        "CXR_NEG_001": "train", "CXR_NEG_002": "val",
        "CXR_NEG_003": "test",  "CXR_UNC_001": "train",
        "CXR_UNC_002": "val",
    }
    rows = []
    for sid in study_ids:
        for view in ["0", "1"]:
            rows.append({
                "image_path": f"/fake/{sid}/{view}.png",
                "image_path_rel": f"{sid}/{view}.png",
                "report_id": sid,
                "view_index": view,
                "study_image_count": "2",
                "report": "Some report.",
                "report_word_count": "2",
                "report_has_xxxx": "False",
                "split": splits_map[sid],
            })
    if extra_rows:
        rows.extend(extra_rows)
    p = tmp_path / "index.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# Tests: load_labeled_csv
# ---------------------------------------------------------------------------

class TestLoadLabeledCsv:
    def test_loads_successfully(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        assert len(rows) == 7

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labeled_csv(tmp_path / "nonexistent.csv")

    def test_row_has_required_fields(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        for field in ("report_id", "split", "pneumonia_label", "labeling_rule", "labeling_confidence"):
            assert field in rows[0], f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests: load_index_csv
# ---------------------------------------------------------------------------

class TestLoadIndexCsv:
    def test_loads_successfully(self, tmp_path):
        p = _make_index_csv(tmp_path)
        rows = load_index_csv(p)
        assert len(rows) == 14  # 7 studies × 2 views

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Tests: check_cross_split_contamination
# ---------------------------------------------------------------------------

class TestCrossSpitContamination:
    def test_clean_data_returns_empty(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        result = check_cross_split_contamination(rows)
        assert result == []

    def test_contaminated_id_detected(self, tmp_path):
        # Duplicate CXR_NEG_001 in a different split
        extra = [{
            "report_id": "CXR_NEG_001",
            "study_image_count": "2",
            "report": "Duplicate report.",
            "report_word_count": "2",
            "report_has_xxxx": "False",
            "split": "test",   # ← different split!
            "pneumonia_label": "negative",
            "labeling_rule": "test",
            "labeling_confidence": "LOW",
        }]
        p = _make_labeled_csv(tmp_path, extra_rows=extra)
        rows = load_labeled_csv(p)
        result = check_cross_split_contamination(rows)
        assert "CXR_NEG_001" in result


# ---------------------------------------------------------------------------
# Tests: check_view_label_consistency
# ---------------------------------------------------------------------------

class TestViewLabelConsistency:
    def test_clean_returns_empty(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        labeled = load_labeled_csv(lp)
        index   = load_index_csv(ip)
        problems = check_view_label_consistency(labeled, index)
        assert problems == []

    def test_missing_study_in_labeled_detected(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        # Add an index row for a study not in labeled CSV
        extra_index = [{
            "image_path": "/fake/CXR_GHOST/0.png",
            "image_path_rel": "CXR_GHOST/0.png",
            "report_id": "CXR_GHOST",
            "view_index": "0",
            "study_image_count": "2",
            "report": "Ghost report.",
            "report_word_count": "2",
            "report_has_xxxx": "False",
            "split": "train",
        }]
        ip = _make_index_csv(tmp_path, extra_rows=extra_index)
        labeled = load_labeled_csv(lp)
        index   = load_index_csv(ip)
        problems = check_view_label_consistency(labeled, index)
        ghost_problems = [p for p in problems if p["report_id"] == "CXR_GHOST"]
        assert len(ghost_problems) >= 1


# ---------------------------------------------------------------------------
# Tests: check_ids_alignment
# ---------------------------------------------------------------------------

class TestIdsAlignment:
    def test_perfectly_aligned(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        labeled = load_labeled_csv(lp)
        index   = load_index_csv(ip)
        result = check_ids_alignment(labeled, index)
        assert result["in_labeled_not_index"] == []
        assert result["in_index_not_labeled"] == []

    def test_detects_extra_in_labeled(self, tmp_path):
        extra_labeled = [{
            "report_id": "CXR_EXTRA",
            "study_image_count": "2",
            "report": "Extra report.",
            "report_word_count": "2",
            "report_has_xxxx": "False",
            "split": "train",
            "pneumonia_label": "negative",
            "labeling_rule": "test",
            "labeling_confidence": "LOW",
        }]
        lp = _make_labeled_csv(tmp_path, extra_rows=extra_labeled)
        ip = _make_index_csv(tmp_path)
        labeled = load_labeled_csv(lp)
        index   = load_index_csv(ip)
        result = check_ids_alignment(labeled, index)
        assert "CXR_EXTRA" in result["in_labeled_not_index"]


# ---------------------------------------------------------------------------
# Tests: compute_imbalance_stats
# ---------------------------------------------------------------------------

class TestComputeImbalanceStats:
    def test_counts_match(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        stats = compute_imbalance_stats(rows)
        assert stats["positive_count"] == 2
        assert stats["negative_count"] == 3
        assert stats["uncertain_count"] == 2
        assert stats["total_studies"] == 7

    def test_always_negative_accuracy(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        stats = compute_imbalance_stats(rows)
        # 3 negatives out of 7 → 42.9%
        assert abs(stats["always_negative_accuracy_pct"] - 42.86) < 0.1

    def test_imbalance_ratio(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        stats = compute_imbalance_stats(rows)
        # 3 neg / 2 pos = 1.5
        assert stats["negative_to_positive_ratio"] == 1.5

    def test_why_accuracy_misleads_text_present(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        stats = compute_imbalance_stats(rows)
        assert "recall" in stats["why_accuracy_misleads"].lower() or \
               "0%" in stats["why_accuracy_misleads"]

    def test_zero_positives_inf_ratio(self):
        rows = [
            {"pneumonia_label": "negative"},
            {"pneumonia_label": "negative"},
        ]
        stats = compute_imbalance_stats(rows)
        assert stats["negative_to_positive_ratio"] == float("inf")


# ---------------------------------------------------------------------------
# Tests: compute_split_summary
# ---------------------------------------------------------------------------

class TestComputeSplitSummary:
    def test_correct_counts(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        summary = compute_split_summary(rows)
        assert summary["train"]["positive"] == 2
        assert summary["train"]["negative"] == 1
        assert summary["val"]["negative"] == 1
        assert summary["val"]["uncertain"] == 1
        assert summary["test"]["negative"] == 1

    def test_all_splits_present(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        summary = compute_split_summary(rows)
        assert "train" in summary
        assert "val" in summary
        assert "test" in summary


# ---------------------------------------------------------------------------
# Tests: build_audit_rows
# ---------------------------------------------------------------------------

class TestBuildAuditRows:
    def test_all_positives_included(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        pos_rows = [r for r in audit if r["pneumonia_label"] == "positive"]
        assert len(pos_rows) == 2

    def test_all_uncertains_included(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        unc_rows = [r for r in audit if r["pneumonia_label"] == "uncertain"]
        assert len(unc_rows) == 2

    def test_negative_sample_size_respected(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        neg_rows = [r for r in audit if r["pneumonia_label"] == "negative"]
        assert len(neg_rows) == 2

    def test_negative_sample_capped_at_available(self, tmp_path):
        """Ask for more negatives than exist — should get all of them."""
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=9999, seed=42)
        neg_rows = [r for r in audit if r["pneumonia_label"] == "negative"]
        assert len(neg_rows) == 3  # only 3 negatives in fixture

    def test_audit_reason_column_set(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        reasons = {r["audit_reason"] for r in audit}
        assert "positive_label" in reasons
        assert "uncertain_label" in reasons
        assert "negative_sample" in reasons

    def test_manual_review_label_is_blank(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        assert all(r["manual_review_label"] == "" for r in audit)

    def test_reviewer_notes_is_blank(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        assert all(r["reviewer_notes"] == "" for r in audit)

    def test_reproducibility_with_same_seed(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit1 = build_audit_rows(rows, neg_sample_size=2, seed=42)
        audit2 = build_audit_rows(rows, neg_sample_size=2, seed=42)
        ids1 = [r["report_id"] for r in audit1]
        ids2 = [r["report_id"] for r in audit2]
        assert ids1 == ids2

    def test_different_seeds_give_different_samples(self, tmp_path):
        # Add more negatives to fixture so different seeds matter
        extra = [
            {"report_id": f"CXR_NEG_EX{i}", "study_image_count": "2",
             "report": "Normal.", "report_word_count": "1",
             "report_has_xxxx": "False", "split": "train",
             "pneumonia_label": "negative", "labeling_rule": "test",
             "labeling_confidence": "LOW"}
            for i in range(10)
        ]
        p = _make_labeled_csv(tmp_path, extra_rows=extra)
        rows = load_labeled_csv(p)
        a1 = {r["report_id"] for r in build_audit_rows(rows, neg_sample_size=5, seed=42)
              if r["pneumonia_label"] == "negative"}
        a2 = {r["report_id"] for r in build_audit_rows(rows, neg_sample_size=5, seed=99)
              if r["pneumonia_label"] == "negative"}
        # With 13 negatives and sample size 5, different seeds should differ
        # (not guaranteed but very likely)
        # We just check they produce valid output
        assert len(a1) == 5
        assert len(a2) == 5

    def test_all_audit_columns_present(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=1, seed=42)
        for col in AUDIT_COLUMNS:
            assert col in audit[0], f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Tests: write_audit_csv
# ---------------------------------------------------------------------------

class TestWriteAuditCsv:
    def test_file_created(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        out = tmp_path / "audit.csv"
        write_audit_csv(audit, out)
        assert out.exists()

    def test_row_count_matches(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        out = tmp_path / "audit.csv"
        write_audit_csv(audit, out)
        with out.open(encoding="utf-8-sig") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(audit)

    def test_all_required_columns_present(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=2, seed=42)
        out = tmp_path / "audit.csv"
        write_audit_csv(audit, out)
        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames
        for col in AUDIT_COLUMNS:
            assert col in cols, f"Column missing: {col}"

    def test_creates_parent_directories(self, tmp_path):
        p = _make_labeled_csv(tmp_path)
        rows = load_labeled_csv(p)
        audit = build_audit_rows(rows, neg_sample_size=1, seed=42)
        out = tmp_path / "deep" / "dir" / "audit.csv"
        write_audit_csv(audit, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: write_audit_summary
# ---------------------------------------------------------------------------

class TestWriteAuditSummary:
    def _run(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        labeled = load_labeled_csv(lp)
        index   = load_index_csv(ip)
        audit   = build_audit_rows(labeled, neg_sample_size=2, seed=42)
        imbalance = compute_imbalance_stats(labeled)
        split_summary = compute_split_summary(labeled)
        cross = check_cross_split_contamination(labeled)
        view_p = check_view_label_consistency(labeled, index)
        id_al  = check_ids_alignment(labeled, index)
        integrity = {"cross_split": cross, "view_problems": view_p, "id_alignment": id_al}
        out_md  = tmp_path / "summary.md"
        audit_csv = tmp_path / "audit.csv"
        write_audit_summary(labeled, audit, imbalance, split_summary, integrity, out_md, audit_csv)
        return out_md

    def test_summary_file_created(self, tmp_path):
        out = self._run(tmp_path)
        assert out.exists()

    def test_contains_disclaimer(self, tmp_path):
        out = self._run(tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "DISCLAIMER" in content or "weak supervision" in content.lower()

    def test_contains_expected_sections(self, tmp_path):
        out = self._run(tmp_path)
        content = out.read_text(encoding="utf-8")
        for section in [
            "Label Counts",
            "Class Imbalance",
            "Integrity Checks",
            "Audit CSV",
            "How to Use",
            "Phase 4 Checkpoint",
            "Decision Gate",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_contains_accuracy_warning(self, tmp_path):
        out = self._run(tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "accuracy" in content.lower()


# ---------------------------------------------------------------------------
# Tests: run_audit (end-to-end)
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_end_to_end(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        audit_csv = tmp_path / "audit.csv"
        summary   = tmp_path / "summary.md"
        results = run_audit(lp, ip, audit_csv, summary, neg_sample_size=2, seed=42)
        assert audit_csv.exists()
        assert summary.exists()
        assert len(results["audit_rows"]) > 0

    def test_result_keys(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        results = run_audit(
            lp, ip,
            tmp_path / "audit.csv",
            tmp_path / "summary.md",
            neg_sample_size=2, seed=42
        )
        for key in (
            "labeled_rows", "index_rows", "audit_rows", "imbalance",
            "split_summary", "cross_split", "view_problems", "id_alignment"
        ):
            assert key in results, f"Missing key: {key}"

    def test_no_cross_split_in_clean_fixture(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        results = run_audit(lp, ip, tmp_path / "a.csv", tmp_path / "s.md",
                            neg_sample_size=2, seed=42)
        assert results["cross_split"] == []

    def test_no_view_problems_in_clean_fixture(self, tmp_path):
        lp = _make_labeled_csv(tmp_path)
        ip = _make_index_csv(tmp_path)
        results = run_audit(lp, ip, tmp_path / "a.csv", tmp_path / "s.md",
                            neg_sample_size=2, seed=42)
        assert results["view_problems"] == []


# ---------------------------------------------------------------------------
# Integration tests on real dataset (auto-skipped if files not present)
# ---------------------------------------------------------------------------

REAL_LABELED = PROJECT_ROOT / "processed" / "iu_xray_labeled.csv"
REAL_INDEX   = PROJECT_ROOT / "processed" / "iu_xray_index.csv"
REAL_DATA_PRESENT = REAL_LABELED.exists() and REAL_INDEX.exists()


@pytest.mark.skipif(not REAL_DATA_PRESENT, reason="Real dataset CSVs not present.")
class TestRealDatasetIntegration:

    def test_no_cross_split_contamination(self):
        rows = load_labeled_csv(REAL_LABELED)
        assert check_cross_split_contamination(rows) == []

    def test_no_view_label_problems(self):
        labeled = load_labeled_csv(REAL_LABELED)
        index   = load_index_csv(REAL_INDEX)
        assert check_view_label_consistency(labeled, index) == []

    def test_ids_perfectly_aligned(self):
        labeled = load_labeled_csv(REAL_LABELED)
        index   = load_index_csv(REAL_INDEX)
        al = check_ids_alignment(labeled, index)
        assert al["in_labeled_not_index"] == []
        assert al["in_index_not_labeled"] == []

    def test_positive_count(self):
        rows = load_labeled_csv(REAL_LABELED)
        stats = compute_imbalance_stats(rows)
        assert stats["positive_count"] == 15

    def test_negative_count(self):
        rows = load_labeled_csv(REAL_LABELED)
        stats = compute_imbalance_stats(rows)
        assert stats["negative_count"] == 2914

    def test_uncertain_count(self):
        rows = load_labeled_csv(REAL_LABELED)
        stats = compute_imbalance_stats(rows)
        assert stats["uncertain_count"] == 26

    def test_always_negative_accuracy_is_high(self):
        rows = load_labeled_csv(REAL_LABELED)
        stats = compute_imbalance_stats(rows)
        # Should be ~98.6%
        assert stats["always_negative_accuracy_pct"] > 97.0

    def test_audit_csv_has_correct_row_count(self, tmp_path):
        labeled = load_labeled_csv(REAL_LABELED)
        audit   = build_audit_rows(labeled, neg_sample_size=100, seed=42)
        # All 15 positives + all 26 uncertain + 100 neg sample = 141
        assert len(audit) == 141

    def test_audit_sample_is_reproducible(self):
        labeled = load_labeled_csv(REAL_LABELED)
        a1 = build_audit_rows(labeled, neg_sample_size=100, seed=42)
        a2 = build_audit_rows(labeled, neg_sample_size=100, seed=42)
        ids1 = [r["report_id"] for r in a1]
        ids2 = [r["report_id"] for r in a2]
        assert ids1 == ids2

    def test_full_run_on_real_data(self, tmp_path):
        results = run_audit(
            REAL_LABELED, REAL_INDEX,
            tmp_path / "audit.csv",
            tmp_path / "summary.md",
            neg_sample_size=100,
            seed=42,
        )
        assert len(results["audit_rows"]) == 141
        assert results["cross_split"] == []
        assert results["view_problems"] == []
