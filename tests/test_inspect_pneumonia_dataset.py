"""
tests/test_inspect_pneumonia_dataset.py
=========================================
Unit and integration tests for src/inspect_pneumonia_dataset.py.

All unit tests use a minimal synthetic fixture written to a temp directory.
Integration tests run against the real dataset and are auto-skipped if it
is not found.

Run with:
    pytest tests/test_inspect_pneumonia_dataset.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inspect_pneumonia_dataset import (
    SPLITS,
    check_image_readability,
    check_inner_duplicate_folder,
    count_images,
    count_pneumonia_subtypes,
    find_dataset_root,
    find_duplicate_filenames,
    find_unexpected_files,
    run_inspection,
    sample_image_dimensions,
    write_inspection_report,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _build_fake_dataset(
    tmp_path: Path,
    counts: dict | None = None,
    add_inner_folder: bool = False,
    add_unexpected: bool = False,
    add_duplicate: bool = False,
) -> Path:
    """
    Build a minimal fake dataset under tmp_path.

    Default layout:
        train/NORMAL/     3 × IM-000N.jpeg
        train/PNEUMONIA/  3 × person1_bacteria_1.jpeg
                          2 × person2_virus_2.jpeg
        val/NORMAL/       1 × IM-0001.jpeg
        val/PNEUMONIA/    1 × person3_bacteria_3.jpeg
        test/NORMAL/      2 × IM-0002.jpeg
        test/PNEUMONIA/   2 × person4_bacteria_4.jpeg

    Parameters
    ----------
    tmp_path         : pytest temp directory
    counts           : override default file counts per split/class
    add_inner_folder : create a nested tmp_path/<name>/ artefact
    add_unexpected   : add a .txt file to train/NORMAL/
    add_duplicate    : add a filename present in two splits
    """
    # Minimal 1×1 white JPEG bytes (valid but trivially small)
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

    default_counts = {
        "train": {"NORMAL": 3, "PNEUMONIA_bacteria": 3, "PNEUMONIA_virus": 2},
        "val":   {"NORMAL": 1, "PNEUMONIA_bacteria": 1, "PNEUMONIA_virus": 0},
        "test":  {"NORMAL": 2, "PNEUMONIA_bacteria": 2, "PNEUMONIA_virus": 0},
    }
    if counts is not None:
        default_counts = counts

    root = tmp_path / "chest_xray"
    root.mkdir()

    # Use a global counter so filenames are unique across all splits/classes
    _counter = [0]

    def _next() -> int:
        _counter[0] += 1
        return _counter[0]

    for split, spec in default_counts.items():
        (root / split / "NORMAL").mkdir(parents=True)
        (root / split / "PNEUMONIA").mkdir(parents=True)

        for _ in range(spec.get("NORMAL", 0)):
            n = _next()
            (root / split / "NORMAL" / f"IM-{n:04d}.jpeg").write_bytes(_TINY_JPEG)

        for _ in range(spec.get("PNEUMONIA_bacteria", 0)):
            n = _next()
            (root / split / "PNEUMONIA" / f"person{n}_bacteria_{n+1}.jpeg").write_bytes(_TINY_JPEG)

        for _ in range(spec.get("PNEUMONIA_virus", 0)):
            n = _next()
            (root / split / "PNEUMONIA" / f"person{n}_virus_{n+100}.jpeg").write_bytes(_TINY_JPEG)

    if add_inner_folder:
        inner = root / "chest_xray"
        inner.mkdir()
        (inner / "train").mkdir()

    if add_unexpected:
        (root / "train" / "NORMAL" / "metadata.txt").write_text("some text")

    if add_duplicate:
        # Copy the first train/NORMAL file into test/NORMAL with the exact same name,
        # creating a genuine cross-split filename duplicate.
        src_file = sorted((root / "train" / "NORMAL").glob("*.jpeg"))[0]
        (root / "test" / "NORMAL" / src_file.name).write_bytes(src_file.read_bytes())

    return root


# ---------------------------------------------------------------------------
# Tests: find_dataset_root
# ---------------------------------------------------------------------------

class TestFindDatasetRoot:
    def test_finds_valid_root(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        found = find_dataset_root(tmp_path)
        assert (found / "train" / "NORMAL").is_dir()

    def test_raises_when_not_found(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            find_dataset_root(empty)

    def test_returns_root_when_start_is_root(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        found = find_dataset_root(root)
        assert found == root


# ---------------------------------------------------------------------------
# Tests: count_images
# ---------------------------------------------------------------------------

class TestCountImages:
    def test_counts_correct(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        counts = count_images(root)
        assert counts["train"]["NORMAL"] == 3
        assert counts["train"]["PNEUMONIA"] == 5   # 3 bacteria + 2 virus
        assert counts["train"]["total"] == 8
        assert counts["val"]["NORMAL"] == 1
        assert counts["val"]["PNEUMONIA"] == 1
        assert counts["test"]["NORMAL"] == 2
        assert counts["test"]["PNEUMONIA"] == 2

    def test_all_splits_present(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        counts = count_images(root)
        for split in SPLITS:
            assert split in counts
            assert counts[split]["exists"] is True

    def test_missing_split_flagged(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        import shutil
        shutil.rmtree(root / "val")
        counts = count_images(root)
        assert counts["val"]["exists"] is False
        assert counts["val"]["NORMAL"] == 0
        assert counts["val"]["PNEUMONIA"] == 0

    def test_total_equals_sum(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        counts = count_images(root)
        for split in SPLITS:
            assert counts[split]["total"] == \
                   counts[split].get("NORMAL", 0) + counts[split].get("PNEUMONIA", 0)


# ---------------------------------------------------------------------------
# Tests: count_pneumonia_subtypes
# ---------------------------------------------------------------------------

class TestCountPneumoniaSubtypes:
    def test_bacteria_virus_counts(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        st = count_pneumonia_subtypes(root)
        assert st["train"]["bacteria"] == 3
        assert st["train"]["virus"] == 2
        assert st["train"]["unknown"] == 0
        assert st["val"]["bacteria"] == 1
        assert st["val"]["virus"] == 0

    def test_all_splits_present(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        st = count_pneumonia_subtypes(root)
        for split in SPLITS:
            assert split in st

    def test_unknown_incremented_for_other_names(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        # Add a pneumonia image with no bacteria/virus in the name
        (root / "train" / "PNEUMONIA" / "mystery_case_001.jpeg").write_bytes(b"\xff\xd8\xff\xd9")
        st = count_pneumonia_subtypes(root)
        assert st["train"]["unknown"] == 1


# ---------------------------------------------------------------------------
# Tests: find_unexpected_files
# ---------------------------------------------------------------------------

class TestFindUnexpectedFiles:
    def test_clean_dataset_has_no_issues(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        issues = find_unexpected_files(root)
        unexpected_type = [i for i in issues if "unexpected file type" in i["issue"]]
        assert unexpected_type == []

    def test_unexpected_file_detected(self, tmp_path):
        root = _build_fake_dataset(tmp_path, add_unexpected=True)
        issues = find_unexpected_files(root)
        unexpected_type = [i for i in issues if "unexpected file type" in i["issue"]]
        assert len(unexpected_type) == 1
        assert ".txt" in unexpected_type[0]["issue"]

    def test_missing_split_reported(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        import shutil
        shutil.rmtree(root / "val")
        issues = find_unexpected_files(root)
        missing = [i for i in issues if "missing" in i["issue"]]
        assert len(missing) >= 1


# ---------------------------------------------------------------------------
# Tests: find_duplicate_filenames
# ---------------------------------------------------------------------------

class TestFindDuplicateFilenames:
    def test_clean_dataset_no_duplicates(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        result = find_duplicate_filenames(root)
        assert result["within_split_duplicates"] == []
        assert result["cross_split_duplicates"] == []

    def test_cross_split_duplicate_detected(self, tmp_path):
        root = _build_fake_dataset(tmp_path, add_duplicate=True)
        result = find_duplicate_filenames(root)
        assert len(result["cross_split_duplicates"]) >= 1

    def test_total_images_counted(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        counts = count_images(root)
        total_expected = sum(counts[s]["total"] for s in SPLITS)
        result = find_duplicate_filenames(root)
        assert result["total_images_checked"] == total_expected


# ---------------------------------------------------------------------------
# Tests: check_image_readability
# ---------------------------------------------------------------------------

class TestCheckImageReadability:
    def test_all_readable_in_clean_fixture(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        result = check_image_readability(root, sample_size=None, seed=42)
        if not result["pillow_available"]:
            pytest.skip("Pillow not installed")
        assert result["unreadable_count"] == 0

    def test_corrupt_image_detected(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        # Overwrite one valid JPEG with garbage bytes
        corrupt = next((root / "train" / "NORMAL").glob("*.jpeg"))
        corrupt.write_bytes(b"this is not a valid jpeg file at all ####")
        result = check_image_readability(root, sample_size=None, seed=42)
        if not result["pillow_available"]:
            pytest.skip("Pillow not installed")
        assert result["unreadable_count"] >= 1

    def test_returns_graceful_result_without_pillow(self, tmp_path, monkeypatch):
        root = _build_fake_dataset(tmp_path)
        import inspect_pneumonia_dataset as mod
        monkeypatch.setattr(mod, "_PILLOW_AVAILABLE", False)
        result = check_image_readability(root)
        assert result["pillow_available"] is False
        assert result["images_checked"] == 0


# ---------------------------------------------------------------------------
# Tests: check_inner_duplicate_folder
# ---------------------------------------------------------------------------

class TestCheckInnerDuplicateFolder:
    def test_no_inner_folder(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        result = check_inner_duplicate_folder(root)
        assert result["inner_folder_detected"] is False
        assert result["inner_path"] is None

    def test_inner_folder_detected(self, tmp_path):
        root = _build_fake_dataset(tmp_path, add_inner_folder=True)
        result = check_inner_duplicate_folder(root)
        assert result["inner_folder_detected"] is True
        assert result["inner_path"] is not None


# ---------------------------------------------------------------------------
# Tests: write_inspection_report
# ---------------------------------------------------------------------------

class TestWriteInspectionReport:
    def test_report_file_created(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        results = run_inspection(root)
        out = tmp_path / "report.md"
        write_inspection_report(results, out)
        assert out.exists()

    def test_report_contains_key_sections(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        results = run_inspection(root)
        out = tmp_path / "report.md"
        write_inspection_report(results, out)
        content = out.read_text(encoding="utf-8")
        for section in [
            "Image Counts per Split",
            "Pneumonia Sub-type",
            "Class Imbalance",
            "Integrity Checks",
            "Image Properties",
            "Comparison with IU X-ray",
            "Recommended Use",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_creates_parent_dirs(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        results = run_inspection(root)
        out = tmp_path / "deep" / "nested" / "report.md"
        write_inspection_report(results, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: run_inspection (end-to-end)
# ---------------------------------------------------------------------------

class TestRunInspection:
    def test_returns_required_keys(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        results = run_inspection(root)
        for key in (
            "dataset_root", "image_counts", "pneumonia_subtypes",
            "unexpected_files", "duplicates", "readability",
            "dimensions", "inner_folder",
        ):
            assert key in results, f"Missing key: {key}"

    def test_no_issues_in_clean_fixture(self, tmp_path):
        root = _build_fake_dataset(tmp_path)
        results = run_inspection(root)
        assert results["duplicates"]["within_split_duplicates"] == []
        assert results["duplicates"]["cross_split_duplicates"] == []
        unexpected_type = [
            i for i in results["unexpected_files"]
            if "unexpected file type" in i["issue"]
        ]
        assert unexpected_type == []


# ---------------------------------------------------------------------------
# Integration tests: real dataset
# ---------------------------------------------------------------------------

def _find_real_dataset() -> Path | None:
    """Return the real dataset root if it exists."""
    # Search from project root
    start = PROJECT_ROOT
    candidates = [
        start / "archive (7)" / "chest_xray",
        start / "chest_xray",
        start / "data" / "chest_xray",
    ]
    for c in candidates:
        if (c / "train" / "NORMAL").is_dir():
            return c
    return None


REAL_DATASET = _find_real_dataset()


@pytest.mark.skipif(REAL_DATASET is None, reason="Real Chest X-Ray dataset not found.")
class TestRealDatasetIntegration:

    def test_all_splits_exist(self):
        counts = count_images(REAL_DATASET)
        for split in SPLITS:
            assert counts[split]["exists"], f"Split '{split}' folder not found"

    def test_known_image_totals(self):
        counts = count_images(REAL_DATASET)
        # Known counts from the Kermany dataset (outer folder)
        assert counts["train"]["NORMAL"] == 1341
        assert counts["train"]["PNEUMONIA"] == 3875
        assert counts["val"]["NORMAL"] == 8
        assert counts["val"]["PNEUMONIA"] == 8
        assert counts["test"]["NORMAL"] == 234
        assert counts["test"]["PNEUMONIA"] == 390

    def test_total_image_count(self):
        counts = count_images(REAL_DATASET)
        total = sum(counts[s]["total"] for s in SPLITS)
        assert total == 5856

    def test_no_duplicate_filenames(self):
        result = find_duplicate_filenames(REAL_DATASET)
        assert result["within_split_duplicates"] == []
        assert result["cross_split_duplicates"] == []

    def test_no_unexpected_files(self):
        issues = find_unexpected_files(REAL_DATASET)
        unexpected_type = [i for i in issues if "unexpected file type" in i["issue"]]
        assert unexpected_type == []

    def test_pneumonia_subtypes(self):
        st = count_pneumonia_subtypes(REAL_DATASET)
        assert st["train"]["bacteria"] == 2530
        assert st["train"]["virus"] == 1345
        assert st["train"]["unknown"] == 0

    def test_inner_folder_detected(self):
        inner = check_inner_duplicate_folder(REAL_DATASET)
        # The Kaggle archive has a chest_xray/chest_xray/ nesting
        assert inner["inner_folder_detected"] is True

    def test_no_unreadable_images_in_sample(self):
        result = check_image_readability(REAL_DATASET, sample_size=50, seed=42)
        if not result["pillow_available"]:
            pytest.skip("Pillow not installed")
        assert result["unreadable_count"] == 0

    def test_full_run_produces_valid_results(self, tmp_path):
        results = run_inspection(REAL_DATASET)
        out = tmp_path / "report.md"
        write_inspection_report(results, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "5,856" in content  # total image count
        assert "Integrity Checks" in content
