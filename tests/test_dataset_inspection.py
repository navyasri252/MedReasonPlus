"""
tests/test_dataset_inspection.py
=================================
Unit tests for src/inspect_dataset.py – Phase 1 of MedReason++.

These tests use a **minimal synthetic fixture** (a tiny fake annotation.json
and a few dummy PNG files written to a temp directory) so they run fast and
require no access to the real 1 GB dataset.

A separate integration-style test at the bottom optionally exercises the real
dataset if it can be found on disk.

Run with pytest from the project root:
    pytest tests/test_dataset_inspection.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make sure src/ is importable regardless of where pytest is invoked from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inspect_dataset import (
    analyse_reports,
    check_image_annotation_links,
    count_disk_images,
    detect_duplicates,
    explain_image_linking,
    inspect_record_fields,
    load_annotation,
    run_full_inspection,
    summarise_splits,
    write_markdown_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_dataset(tmp_path: Path) -> Path:
    """
    Build a minimal fake IU X-ray dataset under `tmp_path`:

        tmp_path/
            annotation.json
            images/
                CXR001_IM-0001/
                    0.png
                    1.png
                CXR002_IM-0002/
                    0.png
                    1.png
                    2.png        ← extra view (tests >2 PNGs per dir)
                CXR003_IM-0003/
                    0.png
                    1.png

    annotation.json references only CXR001 (train), CXR002 (val), CXR003 (test).
    CXR002 gets 2 PNG references but has 3 files → tests orphan detection.
    """
    # Create image files
    studies = {
        "CXR001_IM-0001": ["0.png", "1.png"],
        "CXR002_IM-0002": ["0.png", "1.png", "2.png"],  # extra view
        "CXR003_IM-0003": ["0.png", "1.png"],
    }
    images_dir = tmp_path / "images"
    for study_id, files in studies.items():
        study_dir = images_dir / study_id
        study_dir.mkdir(parents=True)
        for fname in files:
            (study_dir / fname).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    # Create annotation.json  (CXR002 annotation only references 0.png and 1.png)
    annotation = {
        "train": [
            {
                "id": "CXR001_IM-0001",
                "report": "The heart size is normal. Lungs are clear.",
                "image_path": ["CXR001_IM-0001/0.png", "CXR001_IM-0001/1.png"],
                "split": "train",
            }
        ],
        "val": [
            {
                "id": "CXR002_IM-0002",
                "report": "XXXX cardiac silhouette. No effusion.",
                "image_path": ["CXR002_IM-0002/0.png", "CXR002_IM-0002/1.png"],
                "split": "val",
            }
        ],
        "test": [
            {
                "id": "CXR003_IM-0003",
                "report": "No acute cardiopulmonary process.",
                "image_path": ["CXR003_IM-0003/0.png", "CXR003_IM-0003/1.png"],
                "split": "test",
            }
        ],
    }
    ann_path = tmp_path / "annotation.json"
    ann_path.write_text(json.dumps(annotation), encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests for load_annotation
# ---------------------------------------------------------------------------

class TestLoadAnnotation:
    def test_loads_successfully(self, fake_dataset):
        data = load_annotation(fake_dataset)
        assert isinstance(data, dict)

    def test_has_three_splits(self, fake_dataset):
        data = load_annotation(fake_dataset)
        assert set(data.keys()) == {"train", "val", "test"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_annotation(tmp_path)  # no annotation.json here


# ---------------------------------------------------------------------------
# Tests for summarise_splits
# ---------------------------------------------------------------------------

class TestSummariseSplits:
    def test_correct_counts(self, fake_dataset):
        data = load_annotation(fake_dataset)
        summary = summarise_splits(data)
        assert summary["train"] == 1
        assert summary["val"] == 1
        assert summary["test"] == 1
        assert summary["total"] == 3

    def test_total_equals_sum_of_splits(self, fake_dataset):
        data = load_annotation(fake_dataset)
        summary = summarise_splits(data)
        total = summary.pop("total")
        assert total == sum(summary.values())


# ---------------------------------------------------------------------------
# Tests for inspect_record_fields
# ---------------------------------------------------------------------------

class TestInspectRecordFields:
    def test_known_keys_present(self, fake_dataset):
        data = load_annotation(fake_dataset)
        fi = inspect_record_fields(data)
        for key in ["id", "report", "image_path", "split"]:
            assert key in fi["record_keys"]

    def test_no_empty_reports_in_fixture(self, fake_dataset):
        data = load_annotation(fake_dataset)
        fi = inspect_record_fields(data)
        assert fi["empty_report_count"] == 0

    def test_two_images_per_record(self, fake_dataset):
        data = load_annotation(fake_dataset)
        fi = inspect_record_fields(data)
        # All fixture records have exactly 2 image paths
        assert fi["images_per_record_distribution"] == {2: 3}

    def test_empty_report_detected(self, fake_dataset):
        """A record with an empty report string should be counted."""
        data = load_annotation(fake_dataset)
        data["train"].append(
            {
                "id": "CXR999_IM-9999",
                "report": "",
                "image_path": ["CXR999_IM-9999/0.png"],
                "split": "train",
            }
        )
        fi = inspect_record_fields(data)
        assert fi["empty_report_count"] == 1
        assert "CXR999_IM-9999" in fi["empty_report_ids"]


# ---------------------------------------------------------------------------
# Tests for analyse_reports
# ---------------------------------------------------------------------------

class TestAnalyseReports:
    def test_word_count_min_max(self, fake_dataset):
        data = load_annotation(fake_dataset)
        rs = analyse_reports(data)
        # Fixture reports have 7, 4, 5 words
        assert rs["report_word_count_min"] >= 1
        assert rs["report_word_count_max"] >= rs["report_word_count_min"]

    def test_xxxx_detection(self, fake_dataset):
        data = load_annotation(fake_dataset)
        rs = analyse_reports(data)
        # Fixture val record contains "XXXX cardiac silhouette."
        assert rs["reports_with_xxxx_redaction"] == 1

    def test_no_xxxx_when_clean(self):
        """No XXXX in clean data → count should be 0."""
        data = {
            "train": [{"id": "X", "report": "Normal chest X-ray.", "image_path": [], "split": "train"}],
            "val": [],
            "test": [],
        }
        rs = analyse_reports(data)
        assert rs["reports_with_xxxx_redaction"] == 0


# ---------------------------------------------------------------------------
# Tests for count_disk_images
# ---------------------------------------------------------------------------

class TestCountDiskImages:
    def test_total_png_count(self, fake_dataset):
        di = count_disk_images(fake_dataset)
        # 2 + 3 + 2 = 7 PNGs
        assert di["total_png_files"] == 7

    def test_subdir_count(self, fake_dataset):
        di = count_disk_images(fake_dataset)
        assert di["total_subdirectories"] == 3

    def test_pngs_per_subdir_distribution(self, fake_dataset):
        di = count_disk_images(fake_dataset)
        dist = di["pngs_per_subdir_distribution"]
        # 2 dirs with 2 PNGs, 1 dir with 3 PNGs
        assert dist.get(2) == 2
        assert dist.get(3) == 1

    def test_missing_images_dir(self, tmp_path):
        di = count_disk_images(tmp_path)  # no images/ sub-dir
        assert "error" in di


# ---------------------------------------------------------------------------
# Tests for check_image_annotation_links
# ---------------------------------------------------------------------------

class TestCheckImageAnnotationLinks:
    def test_no_missing_files(self, fake_dataset):
        data = load_annotation(fake_dataset)
        lc = check_image_annotation_links(data, fake_dataset)
        assert lc["missing_from_disk"] == 0

    def test_orphan_count(self, fake_dataset):
        """CXR002 has 3 PNGs on disk but annotation only references 2 → 1 orphan."""
        data = load_annotation(fake_dataset)
        lc = check_image_annotation_links(data, fake_dataset)
        assert lc["orphaned_on_disk_not_in_annotation"] == 1

    def test_total_referenced_paths(self, fake_dataset):
        data = load_annotation(fake_dataset)
        lc = check_image_annotation_links(data, fake_dataset)
        # 3 records × 2 paths each = 6
        assert lc["total_referenced_image_paths"] == 6

    def test_missing_file_detected(self, fake_dataset):
        """Add a record pointing to a non-existent file → missing count rises."""
        data = load_annotation(fake_dataset)
        data["train"].append(
            {
                "id": "CXR_GHOST",
                "report": "Ghost report.",
                "image_path": ["CXR_GHOST/0.png"],
                "split": "train",
            }
        )
        lc = check_image_annotation_links(data, fake_dataset)
        assert lc["missing_from_disk"] == 1


# ---------------------------------------------------------------------------
# Tests for detect_duplicates
# ---------------------------------------------------------------------------

class TestDetectDuplicates:
    def test_no_duplicates_in_clean_fixture(self, fake_dataset):
        data = load_annotation(fake_dataset)
        dc = detect_duplicates(data)
        assert dc["duplicate_record_ids"] == 0
        assert dc["cross_split_duplicates"] == 0

    def test_duplicate_within_split_detected(self, fake_dataset):
        data = load_annotation(fake_dataset)
        # Add the same record again to training split
        data["train"].append(data["train"][0].copy())
        dc = detect_duplicates(data)
        assert dc["duplicate_record_ids"] >= 1

    def test_cross_split_duplicate_detected(self, fake_dataset):
        data = load_annotation(fake_dataset)
        # Put the train record into val as well
        data["val"].append(data["train"][0].copy())
        dc = detect_duplicates(data)
        assert dc["cross_split_duplicates"] >= 1


# ---------------------------------------------------------------------------
# Tests for explain_image_linking
# ---------------------------------------------------------------------------

class TestExplainImageLinking:
    def test_zero_mismatches_in_clean_fixture(self, fake_dataset):
        data = load_annotation(fake_dataset)
        ln = explain_image_linking(data)
        assert ln["id_subdir_mismatches"] == 0

    def test_mismatch_detected(self):
        """A record whose image_path subdir differs from id → mismatch."""
        data = {
            "train": [
                {
                    "id": "CXR001_IM-0001",
                    "report": "Normal.",
                    "image_path": ["WRONG_DIR/0.png"],
                    "split": "train",
                }
            ],
            "val": [],
            "test": [],
        }
        ln = explain_image_linking(data)
        assert ln["id_subdir_mismatches"] == 1

    def test_records_checked_count(self, fake_dataset):
        data = load_annotation(fake_dataset)
        ln = explain_image_linking(data)
        assert ln["records_checked"] == 3


# ---------------------------------------------------------------------------
# Tests for write_markdown_report
# ---------------------------------------------------------------------------

class TestWriteMarkdownReport:
    def test_file_is_created(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        results = {
            "data_root": str(fake_dataset),
            "split_summary": summarise_splits(data),
            "field_info": inspect_record_fields(data),
            "report_stats": analyse_reports(data),
            "disk_info": count_disk_images(fake_dataset),
            "link_check": check_image_annotation_links(data, fake_dataset),
            "dup_check": detect_duplicates(data),
            "linking": explain_image_linking(data),
        }
        out_path = tmp_path / "out" / "report.md"
        write_markdown_report(results, out_path)
        assert out_path.exists()

    def test_markdown_contains_key_sections(self, fake_dataset, tmp_path):
        data = load_annotation(fake_dataset)
        results = {
            "data_root": str(fake_dataset),
            "split_summary": summarise_splits(data),
            "field_info": inspect_record_fields(data),
            "report_stats": analyse_reports(data),
            "disk_info": count_disk_images(fake_dataset),
            "link_check": check_image_annotation_links(data, fake_dataset),
            "dup_check": detect_duplicates(data),
            "linking": explain_image_linking(data),
        }
        out_path = tmp_path / "report.md"
        write_markdown_report(results, out_path)
        content = out_path.read_text(encoding="utf-8")
        for section in [
            "Dataset Inspection Report",
            "Split Summary",
            "Record Fields",
            "Image Files",
            "Report Text Quality",
            "Missing Files",
            "Duplicate Records",
            "Summary",
        ]:
            assert section in content, f"Expected section '{section}' not found in report"


# ---------------------------------------------------------------------------
# Integration test: run the FULL inspection on the real dataset (optional)
# ---------------------------------------------------------------------------

def _find_real_dataset() -> Path | None:
    """Return the real dataset root if it exists on disk, else None."""
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

@pytest.mark.skipif(REAL_DATASET is None, reason="Real IU X-ray dataset not found on disk")
class TestRealDatasetIntegration:
    """
    These tests run only when the full ~1 GB dataset is present.
    They verify the expected counts from the actual dataset.
    """

    def test_total_records(self):
        data = load_annotation(REAL_DATASET)
        summary = summarise_splits(data)
        assert summary["total"] == 2955, f"Expected 2955 records, got {summary['total']}"

    def test_split_counts(self):
        data = load_annotation(REAL_DATASET)
        summary = summarise_splits(data)
        assert summary["train"] == 2069
        assert summary["val"] == 296
        assert summary["test"] == 590

    def test_no_missing_images(self):
        data = load_annotation(REAL_DATASET)
        lc = check_image_annotation_links(data, REAL_DATASET)
        assert lc["missing_from_disk"] == 0

    def test_no_duplicate_records(self):
        data = load_annotation(REAL_DATASET)
        dc = detect_duplicates(data)
        assert dc["duplicate_record_ids"] == 0
        assert dc["cross_split_duplicates"] == 0

    def test_no_id_subdir_mismatches(self):
        data = load_annotation(REAL_DATASET)
        ln = explain_image_linking(data)
        assert ln["id_subdir_mismatches"] == 0

    def test_full_inspection_runs_without_error(self):
        results = run_full_inspection(REAL_DATASET)
        assert results["split_summary"]["total"] == 2955
