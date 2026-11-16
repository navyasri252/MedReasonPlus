"""
tests/test_build_dataset_index.py
==================================
Unit and integration tests for src/build_dataset_index.py — Phase 2.

Design
------
All unit tests use a *minimal synthetic fixture* built in a temp directory.
No access to the real ~1 GB dataset is required for the unit tests.

Integration tests (class TestRealDatasetIntegration) are auto-skipped unless
the real dataset is found on disk.

Run with:
    pytest tests/test_build_dataset_index.py -v
"""

import csv
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_dataset_index import (
    CSV_COLUMNS,
    build_index_rows,
    build_study_image_counts,
    find_dataset_root,
    load_annotation,
    run_build,
    validate_csv,
    write_csv,
    write_summary_report,
)


# ---------------------------------------------------------------------------
# Shared fixture: minimal fake dataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    """
    Create a tiny fake IU X-ray dataset:

        tmp_path/
            annotation.json
            images/
                CXR001_IM-0001/   0.png  1.png
                CXR002_IM-0002/   0.png  1.png  2.png   ← extra view
                CXR003_IM-0003/   0.png  1.png

    Three records: CXR001 → train, CXR002 → val, CXR003 → test.
    CXR002's annotation references only views 0 and 1; view 2 is orphaned.
    """
    studies = {
        "CXR001_IM-0001": ["0.png", "1.png"],
        "CXR002_IM-0002": ["0.png", "1.png", "2.png"],
        "CXR003_IM-0003": ["0.png", "1.png"],
    }
    images_dir = tmp_path / "images"
    for sid, files in studies.items():
        study_dir = images_dir / sid
        study_dir.mkdir(parents=True)
        for fname in files:
            (study_dir / fname).write_bytes(b"\x89PNG\r\n\x1a\n")

    annotation = {
        "train": [{
            "id": "CXR001_IM-0001",
            "report": "Normal heart size. Clear lungs bilaterally.",
            "image_path": ["CXR001_IM-0001/0.png", "CXR001_IM-0001/1.png"],
            "split": "train",
        }],
        "val": [{
            "id": "CXR002_IM-0002",
            "report": "XXXX enlargement of the cardiac silhouette. No effusion.",
            "image_path": ["CXR002_IM-0002/0.png", "CXR002_IM-0002/1.png"],
            "split": "val",
        }],
        "test": [{
            "id": "CXR003_IM-0003",
            "report": "No acute cardiopulmonary process identified.",
            "image_path": ["CXR003_IM-0003/0.png", "CXR003_IM-0003/1.png"],
            "split": "test",
        }],
    }
    (tmp_path / "annotation.json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: find_dataset_root
# ---------------------------------------------------------------------------

class TestFindDatasetRoot:
    def test_finds_real_layout(self, fake_dataset):
        # fake_dataset IS a valid root — should return it (or a subdirectory of it)
        # that contains annotation.json and images/
        found = find_dataset_root(fake_dataset)
        assert (found / "annotation.json").exists()
        assert (found / "images").is_dir()

    def test_raises_when_no_dataset(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            find_dataset_root(empty)


# ---------------------------------------------------------------------------
# Tests: load_annotation
# ---------------------------------------------------------------------------

class TestLoadAnnotation:
    def test_returns_dict_with_splits(self, fake_dataset):
        data = load_annotation(fake_dataset)
        assert isinstance(data, dict)
        assert set(data.keys()) == {"train", "val", "test"}

    def test_record_structure(self, fake_dataset):
        data = load_annotation(fake_dataset)
        rec = data["train"][0]
        for field in ("id", "report", "image_path", "split"):
            assert field in rec, f"Missing field: {field}"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_annotation(tmp_path)


# ---------------------------------------------------------------------------
# Tests: build_study_image_counts
# ---------------------------------------------------------------------------

class TestBuildStudyImageCounts:
    def test_counts_correct(self, fake_dataset):
        counts = build_study_image_counts(fake_dataset)
        assert counts["CXR001_IM-0001"] == 2
        assert counts["CXR002_IM-0002"] == 3   # has extra view
        assert counts["CXR003_IM-0003"] == 2

    def test_all_studies_counted(self, fake_dataset):
        counts = build_study_image_counts(fake_dataset)
        assert len(counts) == 3

    def test_empty_images_dir(self, tmp_path):
        (tmp_path / "images").mkdir()
        counts = build_study_image_counts(tmp_path)
        assert counts == {}


# ---------------------------------------------------------------------------
# Tests: build_index_rows
# ---------------------------------------------------------------------------

class TestBuildIndexRows:
    def test_row_count(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        # 3 records × 2 image paths each = 6 rows
        assert len(rows) == 6

    def test_columns_present(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        for col in CSV_COLUMNS:
            assert col in rows[0], f"Missing column: {col}"

    def test_view_index_values(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        view_indices = {r["view_index"] for r in rows}
        assert view_indices == {0, 1}

    def test_split_values_correct(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        splits_seen = {r["split"] for r in rows}
        assert splits_seen == {"train", "val", "test"}

    def test_xxxx_flag_set(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        # CXR002 report contains XXXX → both its rows should be flagged
        xxxx_rows = [r for r in rows if r["report_id"] == "CXR002_IM-0002"]
        assert all(r["report_has_xxxx"] is True for r in xxxx_rows)

    def test_no_xxxx_flag_clear(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        no_xxxx_rows = [r for r in rows if r["report_id"] == "CXR001_IM-0001"]
        assert all(r["report_has_xxxx"] is False for r in no_xxxx_rows)

    def test_word_count_computed(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        for r in rows:
            expected_wc = len(r["report"].split()) if r["report"].strip() else 0
            assert r["report_word_count"] == expected_wc

    def test_study_image_count_reflected(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        cxr002_rows = [r for r in rows if r["report_id"] == "CXR002_IM-0002"]
        # CXR002 has 3 PNGs on disk
        assert all(r["study_image_count"] == 3 for r in cxr002_rows)

    def test_no_issues_with_clean_fixture(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        _, issues = build_index_rows(data, fake_dataset, counts)
        assert issues == []

    def test_empty_report_flagged(self, fake_dataset):
        data = load_annotation(fake_dataset)
        data["train"].append({
            "id": "CXR_EMPTY",
            "report": "",
            "image_path": ["CXR001_IM-0001/0.png"],
            "split": "train",
        })
        counts = build_study_image_counts(fake_dataset)
        _, issues = build_index_rows(data, fake_dataset, counts)
        empty_issues = [i for i in issues if i["type"] == "empty_report"]
        assert len(empty_issues) == 1
        assert empty_issues[0]["record_id"] == "CXR_EMPTY"

    def test_missing_image_flagged(self, fake_dataset):
        data = load_annotation(fake_dataset)
        data["test"].append({
            "id": "CXR_GHOST",
            "report": "Ghost report text here.",
            "image_path": ["CXR_GHOST/0.png"],
            "split": "test",
        })
        counts = build_study_image_counts(fake_dataset)
        _, issues = build_index_rows(data, fake_dataset, counts)
        missing_issues = [i for i in issues if i["type"] == "missing_image"]
        assert len(missing_issues) == 1
        assert "CXR_GHOST/0.png" in missing_issues[0]["detail"]

    def test_duplicate_id_flagged(self, fake_dataset):
        data = load_annotation(fake_dataset)
        # Add a duplicate of CXR001
        data["val"].append(data["train"][0].copy())
        counts = build_study_image_counts(fake_dataset)
        _, issues = build_index_rows(data, fake_dataset, counts)
        dup_issues = [i for i in issues if i["type"] == "duplicate_id"]
        assert len(dup_issues) >= 1

    def test_image_path_is_absolute(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        for r in rows:
            assert Path(r["image_path"]).is_absolute(), \
                f"image_path should be absolute: {r['image_path']}"

    def test_image_path_rel_matches_annotation(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        # Every image_path_rel should be exactly as listed in annotation
        ann_paths = set()
        for split_records in data.values():
            for rec in split_records:
                for p in rec["image_path"]:
                    ann_paths.add(p)
        row_rels = {r["image_path_rel"] for r in rows}
        assert row_rels == ann_paths


# ---------------------------------------------------------------------------
# Tests: write_csv and validate_csv
# ---------------------------------------------------------------------------

class TestWriteCsvAndValidate:
    def test_csv_file_created(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        out = tmp_path / "out" / "index.csv"
        write_csv(rows, out)
        assert out.exists()

    def test_csv_has_correct_columns(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        out = tmp_path / "index.csv"
        write_csv(rows, out)
        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            actual_cols = reader.fieldnames
        for col in CSV_COLUMNS:
            assert col in actual_cols, f"Column missing from CSV: {col}"

    def test_csv_row_count(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        out = tmp_path / "index.csv"
        write_csv(rows, out)
        with out.open(encoding="utf-8-sig") as fh:
            data_rows = list(csv.DictReader(fh))
        assert len(data_rows) == 6  # 3 studies × 2 views

    def test_validate_passes_on_clean_csv(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, _ = build_index_rows(data, fake_dataset, counts)
        out = tmp_path / "index.csv"
        write_csv(rows, out)
        result = validate_csv(out)
        assert result["rows_read"] == 6
        assert result["missing_image_paths"] == 0
        assert result["empty_report_ids"] == 0
        assert result["empty_report_texts"] == 0
        assert result["column_issues"] == []

    def test_validate_detects_missing_csv(self, tmp_path):
        result = validate_csv(tmp_path / "nonexistent.csv")
        assert "error" in result

    def test_validate_detects_bad_paths(self, fake_dataset, tmp_path):
        """Manually write a CSV with a broken image_path and confirm validation catches it."""
        out = tmp_path / "bad.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "image_path": "/nonexistent/path/0.png",
                "image_path_rel": "CXR_BAD/0.png",
                "report_id": "CXR_BAD",
                "view_index": 0,
                "study_image_count": 2,
                "report": "Some report text.",
                "report_word_count": 3,
                "report_has_xxxx": False,
                "split": "test",
            })
        result = validate_csv(out)
        assert result["missing_image_paths"] == 1


# ---------------------------------------------------------------------------
# Tests: write_summary_report
# ---------------------------------------------------------------------------

class TestWriteSummaryReport:
    def _build_results(self, fake_dataset):
        data = load_annotation(fake_dataset)
        counts = build_study_image_counts(fake_dataset)
        rows, issues = build_index_rows(data, fake_dataset, counts)
        return rows, issues

    def test_summary_file_created(self, fake_dataset, tmp_path):
        rows, issues = self._build_results(fake_dataset)
        out_csv = tmp_path / "index.csv"
        write_csv(rows, out_csv)
        validation = validate_csv(out_csv)
        out_md = tmp_path / "summary.md"
        write_summary_report(rows, issues, validation, fake_dataset, out_csv, out_md)
        assert out_md.exists()

    def test_summary_contains_expected_sections(self, fake_dataset, tmp_path):
        rows, issues = self._build_results(fake_dataset)
        out_csv = tmp_path / "index.csv"
        write_csv(rows, out_csv)
        validation = validate_csv(out_csv)
        out_md = tmp_path / "summary.md"
        write_summary_report(rows, issues, validation, fake_dataset, out_csv, out_md)
        content = out_md.read_text(encoding="utf-8")
        for section in [
            "CSV Schema",
            "Record Counts",
            "View Distribution",
            "Report Text",
            "Issues Log",
            "CSV Validation",
            "Phase 2 Checkpoint",
        ]:
            assert section in content, f"Section '{section}' missing from summary"

    def test_summary_mentions_no_findings_impression(self, fake_dataset, tmp_path):
        """Summary should explicitly state that findings/impression are not available."""
        rows, issues = self._build_results(fake_dataset)
        out_csv = tmp_path / "index.csv"
        write_csv(rows, out_csv)
        validation = validate_csv(out_csv)
        out_md = tmp_path / "summary.md"
        write_summary_report(rows, issues, validation, fake_dataset, out_csv, out_md)
        content = out_md.read_text(encoding="utf-8")
        assert "findings" in content.lower()
        assert "impression" in content.lower()


# ---------------------------------------------------------------------------
# Tests: run_build (end-to-end)
# ---------------------------------------------------------------------------

class TestRunBuild:
    def test_end_to_end_clean(self, fake_dataset, tmp_path):
        csv_out = tmp_path / "index.csv"
        summary_out = tmp_path / "summary.md"
        results = run_build(fake_dataset, csv_out, summary_out)

        assert csv_out.exists()
        assert summary_out.exists()
        assert len(results["rows"]) == 6
        assert results["issues"] == []
        assert results["validation"]["missing_image_paths"] == 0

    def test_results_dict_keys(self, fake_dataset, tmp_path):
        results = run_build(
            fake_dataset,
            tmp_path / "index.csv",
            tmp_path / "summary.md",
        )
        for key in ("data_root", "csv_path", "rows", "issues", "validation"):
            assert key in results, f"Missing key in results: {key}"


# ---------------------------------------------------------------------------
# Integration tests on the real dataset (auto-skipped if not present)
# ---------------------------------------------------------------------------

def _find_real_dataset() -> Path | None:
    candidates = [
        PROJECT_ROOT / "iu_xray" / "iu_xray",
        PROJECT_ROOT / "data" / "iu_xray",
        PROJECT_ROOT / "data" / "iu_xray" / "iu_xray",
    ]
    for c in candidates:
        if (c / "annotation.json").exists() and (c / "images").is_dir():
            return c
    return None


REAL_DATASET = _find_real_dataset()


@pytest.mark.skipif(REAL_DATASET is None, reason="Real IU X-ray dataset not found.")
class TestRealDatasetIntegration:
    def test_row_count(self):
        data = load_annotation(REAL_DATASET)
        counts = build_study_image_counts(REAL_DATASET)
        rows, _ = build_index_rows(data, REAL_DATASET, counts)
        # 2955 studies × 2 annotation-referenced views each = 5910 rows
        assert len(rows) == 5910, f"Expected 5910 rows, got {len(rows)}"

    def test_no_missing_images(self):
        data = load_annotation(REAL_DATASET)
        counts = build_study_image_counts(REAL_DATASET)
        _, issues = build_index_rows(data, REAL_DATASET, counts)
        missing = [i for i in issues if i["type"] == "missing_image"]
        assert len(missing) == 0

    def test_no_empty_reports(self):
        data = load_annotation(REAL_DATASET)
        counts = build_study_image_counts(REAL_DATASET)
        _, issues = build_index_rows(data, REAL_DATASET, counts)
        empty = [i for i in issues if i["type"] == "empty_report"]
        assert len(empty) == 0

    def test_no_duplicate_ids(self):
        data = load_annotation(REAL_DATASET)
        counts = build_study_image_counts(REAL_DATASET)
        _, issues = build_index_rows(data, REAL_DATASET, counts)
        dups = [i for i in issues if i["type"] == "duplicate_id"]
        assert len(dups) == 0

    def test_full_build_runs(self, tmp_path):
        csv_out = tmp_path / "index.csv"
        summary_out = tmp_path / "summary.md"
        results = run_build(REAL_DATASET, csv_out, summary_out)
        assert csv_out.exists()
        assert results["validation"]["missing_image_paths"] == 0
        assert results["validation"]["rows_read"] == 5910
