"""
tests/test_label_pneumonia_reports.py
======================================
Unit and integration tests for src/label_pneumonia_reports.py — Phase 3.

Test structure
--------------
TestClassifyReport    — exercises classify_report() directly with synthetic text
TestBuildLabeledRows  — exercises the CSV-to-labeled-rows pipeline via a fake CSV
TestWriteAndVerify    — exercises write_labeled_csv and file integrity
TestSummaryReport     — exercises write_labeling_summary
TestRunLabeling       — end-to-end run_labeling() on a synthetic fixture
TestRealDataset       — integration tests on the real dataset (auto-skipped)

Run with:
    pytest tests/test_label_pneumonia_reports.py -v
"""

import csv
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add src/ to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from label_pneumonia_reports import (
    HIGH,
    LOW,
    MEDIUM,
    NEGATIVE,
    POSITIVE,
    UNCERTAIN,
    build_labeled_rows,
    classify_report,
    run_labeling,
    write_labeled_csv,
    write_labeling_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_of(text: str) -> str:
    """Return just the label string for a report text."""
    label, _, _ = classify_report(text)
    return label


def rule_of(text: str) -> str:
    """Return just the rule name for a report text."""
    _, rule, _ = classify_report(text)
    return rule


def conf_of(text: str) -> str:
    """Return just the confidence string for a report text."""
    _, _, conf = classify_report(text)
    return conf


# ---------------------------------------------------------------------------
# Tests: classify_report — explicit positive cases
# ---------------------------------------------------------------------------

class TestClassifyReportPositive:

    def test_right_lower_lobe_pneumonia(self):
        txt = ("The cardiomediastinal silhouette is normal. "
               "There is a right lower lobe pneumonia. No pleural effusion.")
        assert label_of(txt) == POSITIVE

    def test_airspace_disease_consistent_with_pneumonia(self):
        txt = "Right upper lobe airspace disease consistent with pneumonia given patient history."
        assert label_of(txt) == POSITIVE

    def test_consistent_with_pneumonia_short(self):
        txt = "In the left lower lobe there is airspace disease consistent with pneumonia."
        assert label_of(txt) == POSITIVE

    def test_suggestive_of_rounded_pneumonia(self):
        txt = ("There is a rounded dense opacity in the lateral left midlung zone "
               "most suggestive of a rounded pneumonia.")
        assert label_of(txt) == POSITIVE

    def test_concerning_for_underlying_pneumonia(self):
        txt = ("Interval development of mild patchy airspace opacities within the "
               "posterior aspect of the right upper lobe, concerning for underlying pneumonia.")
        assert label_of(txt) == POSITIVE

    def test_pneumonia_seen_on_ct(self):
        txt = ("Cardiomediastinal silhouette stable. "
               "Pneumonia seen on CT examination dated XXXX (not seen on prior radiograph).")
        assert label_of(txt) == POSITIVE

    def test_left_lower_lobe_airspace_disease_explicit(self):
        txt = ("There is left lower lobe airspace disease identified. "
               "Moderate left pleural effusion.")
        assert label_of(txt) == POSITIVE

    def test_positive_high_confidence_direct(self):
        txt = "There is a right lower lobe pneumonia."
        _, _, conf = classify_report(txt)
        assert conf == HIGH

    def test_positive_medium_confidence_probable(self):
        txt = ("Interval development of mild patchy airspace opacities, "
               "concerning for underlying pneumonia.")
        _, _, conf = classify_report(txt)
        assert conf in (HIGH, MEDIUM)


# ---------------------------------------------------------------------------
# Tests: classify_report — explicit negation cases
# ---------------------------------------------------------------------------

class TestClassifyReportNegative:

    def test_no_pneumonia_simple(self):
        txt = "Heart size normal. Lungs are clear. No pneumonia, effusions, edema, pneumothorax."
        assert label_of(txt) == NEGATIVE

    def test_no_focal_air_space_opacity_to_suggest(self):
        txt = ("The heart, pulmonary XXXX and mediastinum are within normal limits. "
               "There is no pleural effusion or pneumothorax. "
               "There is no focal air space opacity to suggest a pneumonia.")
        assert label_of(txt) == NEGATIVE

    def test_no_focal_airspace_consolidation_to_suggest(self):
        txt = ("Lungs are hyperexpanded. "
               "No focal airspace consolidation to suggest pneumonia. "
               "No pleural effusion or pneumothorax.")
        assert label_of(txt) == NEGATIVE

    def test_no_evidence_of_pneumonia(self):
        txt = ("The lungs appear clear. No evidence of focal pneumonia. "
               "The heart and pulmonary vasculature are normal.")
        assert label_of(txt) == NEGATIVE

    def test_lungs_clear_proxy_negative(self):
        txt = ("Cardiac and mediastinal contours are within normal limits. "
               "The lungs are clear. Bony structures are intact.")
        assert label_of(txt) == NEGATIVE

    def test_no_focal_consolidation_proxy_negative(self):
        txt = ("The lungs are clear bilaterally. "
               "No evidence of focal consolidation, pneumothorax, or pleural effusion. "
               "Cardiomediastinal silhouette is unremarkable.")
        assert label_of(txt) == NEGATIVE

    def test_no_active_pneumonia(self):
        txt = "No active pneumonia. Mild streaky scarring in the right upper lobe. Bony structures intact."
        assert label_of(txt) == NEGATIVE

    def test_grossly_clear_without_pneumonic_consolidation(self):
        txt = "The lungs are grossly clear without focal pneumonic consolidation."
        assert label_of(txt) == NEGATIVE

    def test_free_of_focal_airspace_disease(self):
        txt = ("Heart size and pulmonary vascularity within normal limits. "
               "The lungs are free of focal airspace disease. "
               "No pleural effusion or pneumothorax is seen.")
        assert label_of(txt) == NEGATIVE

    def test_no_airspace_opacities_to_suggest(self):
        txt = ("There are no focal airspace opacities to suggest pneumonia. "
               "The pleural spaces are clear. The heart and pulmonary vasculature are normal.")
        assert label_of(txt) == NEGATIVE

    def test_negative_high_confidence_explicit(self):
        txt = "There is no pneumonia. The lungs are clear."
        _, _, conf = classify_report(txt)
        assert conf == HIGH

    def test_no_pneumonia_does_not_label_positive(self):
        """A report that mentions pneumonia only in negation must NOT be positive."""
        txt = "No pneumonia, no effusions, no pneumothorax."
        assert label_of(txt) == NEGATIVE

    def test_no_x_to_suggest_pneumonia_is_negative(self):
        """The phrase 'to suggest pneumonia' means the radiologist found NO opacity."""
        txt = "There is no focal air space opacity to suggest a pneumonia."
        assert label_of(txt) == NEGATIVE

    def test_no_focal_consolidation_to_suggest_pneumonia_is_negative(self):
        txt = "There is no focal consolidation to suggest pneumonia."
        assert label_of(txt) == NEGATIVE


# ---------------------------------------------------------------------------
# Tests: classify_report — uncertain/hedged cases
# ---------------------------------------------------------------------------

class TestClassifyReportUncertain:

    def test_could_represent_pneumonia(self):
        txt = ("There are streaky retrocardiac left lower lobe opacities; "
               "in the correct clinical setting this could represent a pneumonia. "
               "There is no pneumothorax or pleural effusion.")
        assert label_of(txt) == UNCERTAIN

    def test_may_reflect_atelectasis_or_pneumonia(self):
        txt = ("There is right middle lobe airspace disease, "
               "may reflect atelectasis or pneumonia. "
               "No pleural effusion. No pneumothorax.")
        assert label_of(txt) == UNCERTAIN

    def test_may_represent_early_pneumonia(self):
        txt = ("Vague patchy opacity in the right midlung which "
               "may represent early pneumonia. "
               "No pleural effusions or pneumothoraces.")
        assert label_of(txt) == UNCERTAIN

    def test_atelectasis_versus_pneumonia(self):
        txt = "Question left basilar atelectasis versus infiltrate. No pneumothorax."
        assert label_of(txt) == UNCERTAIN

    def test_pneumonitis(self):
        txt = ("There is patchy opacity in the left base concerning for "
               "atelectasis versus pneumonitis.")
        assert label_of(txt) == UNCERTAIN

    def test_cannot_exclude(self):
        txt = "Cannot exclude early pulmonary consolidation in this limited study."
        assert label_of(txt) == UNCERTAIN

    def test_conflicting_positive_and_negative_gives_uncertain(self):
        """A report asserting pneumonia AND denying it → uncertain."""
        txt = ("There is a right lower lobe pneumonia. "
               "No focal consolidation is seen elsewhere. "
               "There is no focal air space opacity to suggest a pneumonia.")
        assert label_of(txt) == UNCERTAIN

    def test_bibasilar_opacities_could_represent_pneumonia(self):
        txt = ("Low lung volumes. Bibasilar opacities. "
               "The left base opacities could represent early pneumonia or atelectasis.")
        assert label_of(txt) == UNCERTAIN


# ---------------------------------------------------------------------------
# Tests: classify_report — edge cases
# ---------------------------------------------------------------------------

class TestClassifyReportEdgeCases:

    def test_empty_string_returns_uncertain(self):
        label, rule, conf = classify_report("")
        assert label == UNCERTAIN
        assert conf == LOW

    def test_whitespace_only_returns_uncertain(self):
        label, _, _ = classify_report("   \n\t  ")
        assert label == UNCERTAIN

    def test_completely_unrelated_text_returns_negative(self):
        """A chest X-ray that never mentions any lung-related term → negative/LOW."""
        txt = "The bony structures are intact. No acute osseous abnormality."
        assert label_of(txt) == NEGATIVE
        assert conf_of(txt) == LOW

    def test_xxxx_masked_clear_lungs_still_negative(self):
        """XXXX tokens should not disrupt negation detection."""
        txt = "Heart XXXX, mediastinum, XXXX, bony structures and lung XXXX are unremarkable."
        # No pneumonia language present → negative LOW
        assert label_of(txt) == NEGATIVE

    def test_pneumothorax_alone_is_not_pneumonia(self):
        """pneumothorax contains 'pneumo' but is NOT pneumonia."""
        txt = "There is a small right pneumothorax. No pleural effusion."
        # Should NOT be positive for pneumonia
        assert label_of(txt) == NEGATIVE

    def test_case_insensitivity(self):
        """Patterns must fire regardless of capitalisation."""
        txt = "THERE IS NO PNEUMONIA. THE LUNGS ARE CLEAR."
        assert label_of(txt) == NEGATIVE

        txt2 = "There Is A Right Lower Lobe Pneumonia."
        assert label_of(txt2) == POSITIVE

    def test_rule_name_is_a_string(self):
        txt = "No pneumonia, no effusion."
        _, rule, _ = classify_report(txt)
        assert isinstance(rule, str) and len(rule) > 0

    def test_confidence_is_valid_value(self):
        for txt in [
            "There is a right lower lobe pneumonia.",
            "No pneumonia, lungs clear.",
            "Atelectasis versus pneumonia cannot be excluded.",
            "",
        ]:
            _, _, conf = classify_report(txt)
            assert conf in (HIGH, MEDIUM, LOW), f"Unexpected confidence {conf!r} for: {txt!r}"

    def test_label_is_valid_value(self):
        for txt in [
            "There is a right lower lobe pneumonia.",
            "No pneumonia, lungs clear.",
            "Could represent pneumonia.",
            "",
        ]:
            lbl, _, _ = classify_report(txt)
            assert lbl in (POSITIVE, NEGATIVE, UNCERTAIN)


# ---------------------------------------------------------------------------
# Fixture: minimal fake Phase 2 CSV
# ---------------------------------------------------------------------------

def _make_fake_csv(tmp_path: Path) -> Path:
    """
    Create a synthetic iu_xray_index.csv with 6 studies (12 image rows)
    covering positive, negative, and uncertain language.
    """
    csv_path = tmp_path / "iu_xray_index.csv"
    fieldnames = [
        "image_path", "image_path_rel", "report_id", "view_index",
        "study_image_count", "report", "report_word_count",
        "report_has_xxxx", "split",
    ]

    studies = [
        ("CXR_POS_001", "There is a right lower lobe pneumonia. No effusion.", "train"),
        ("CXR_NEG_001", "No pneumonia, effusions, edema, or pneumothorax. Lungs are clear.", "train"),
        ("CXR_NEG_002",
         "There is no focal air space opacity to suggest a pneumonia. Lungs clear.", "val"),
        ("CXR_UNC_001",
         "Left lower lobe opacities; could represent a pneumonia in the right clinical setting.", "val"),
        ("CXR_UNC_002",
         "Atelectasis versus pneumonia cannot be excluded at the left base.", "test"),
        ("CXR_NEG_003",
         "Cardiac and mediastinal contours are within normal limits. Lungs clear.", "test"),
    ]

    rows = []
    for sid, report, split in studies:
        for view in ["0", "1"]:
            rows.append({
                "image_path": f"/fake/images/{sid}/{view}.png",
                "image_path_rel": f"{sid}/{view}.png",
                "report_id": sid,
                "view_index": view,
                "study_image_count": "2",
                "report": report,
                "report_word_count": str(len(report.split())),
                "report_has_xxxx": "False",
                "split": split,
            })

    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ---------------------------------------------------------------------------
# Tests: build_labeled_rows
# ---------------------------------------------------------------------------

class TestBuildLabeledRows:

    def test_one_row_per_study(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        assert len(labeled) == 6  # 6 studies, not 12 image rows

    def test_new_columns_present(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        for col in ("pneumonia_label", "labeling_rule", "labeling_confidence"):
            assert col in fieldnames, f"Missing column: {col}"
            assert all(col in r for r in labeled)

    def test_positive_study_labeled_correctly(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        pos = next(r for r in labeled if r["report_id"] == "CXR_POS_001")
        assert pos["pneumonia_label"] == POSITIVE

    def test_negative_study_labeled_correctly(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        neg = next(r for r in labeled if r["report_id"] == "CXR_NEG_001")
        assert neg["pneumonia_label"] == NEGATIVE

    def test_uncertain_study_labeled_correctly(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        unc = next(r for r in labeled if r["report_id"] == "CXR_UNC_001")
        assert unc["pneumonia_label"] == UNCERTAIN

    def test_no_image_path_columns_in_output(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        # image_path and view_index are per-image — should be removed at study level
        assert "view_index" not in fieldnames
        assert "image_path" not in fieldnames

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_labeled_rows(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Tests: write_labeled_csv
# ---------------------------------------------------------------------------

class TestWriteLabeledCsv:

    def test_file_created(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        out = tmp_path / "labeled.csv"
        write_labeled_csv(labeled, fieldnames, out)
        assert out.exists()

    def test_row_count_matches(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        out = tmp_path / "labeled.csv"
        write_labeled_csv(labeled, fieldnames, out)
        with out.open(encoding="utf-8-sig") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == 6

    def test_columns_written_correctly(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        out = tmp_path / "labeled.csv"
        write_labeled_csv(labeled, fieldnames, out)
        with out.open(encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames
        for col in ("pneumonia_label", "labeling_rule", "labeling_confidence", "report_id"):
            assert col in cols

    def test_creates_parent_directories(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        out = tmp_path / "deep" / "nested" / "labeled.csv"
        write_labeled_csv(labeled, fieldnames, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: write_labeling_summary
# ---------------------------------------------------------------------------

class TestWriteLabelingSummary:

    def test_summary_created(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, fieldnames = build_labeled_rows(csv_path)
        out_md = tmp_path / "summary.md"
        write_labeling_summary(labeled, out_md, csv_path)
        assert out_md.exists()

    def test_summary_has_disclaimer(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        out_md = tmp_path / "summary.md"
        write_labeling_summary(labeled, out_md, csv_path)
        content = out_md.read_text(encoding="utf-8")
        assert "DISCLAIMER" in content or "disclaimer" in content.lower()
        assert "weak supervision" in content.lower() or "weak" in content.lower()

    def test_summary_contains_expected_sections(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        out_md = tmp_path / "summary.md"
        write_labeling_summary(labeled, out_md, csv_path)
        content = out_md.read_text(encoding="utf-8")
        for section in [
            "Label Counts",
            "Confidence Breakdown",
            "Distribution by Split",
            "Representative Examples",
            "Labeling Methodology",
            "Phase 3 Checkpoint",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_summary_mentions_all_three_labels(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        labeled, _ = build_labeled_rows(csv_path)
        out_md = tmp_path / "summary.md"
        write_labeling_summary(labeled, out_md, csv_path)
        content = out_md.read_text(encoding="utf-8")
        for lbl in (POSITIVE, NEGATIVE, UNCERTAIN):
            assert lbl in content, f"Label '{lbl}' not mentioned in summary"


# ---------------------------------------------------------------------------
# Tests: run_labeling (end-to-end)
# ---------------------------------------------------------------------------

class TestRunLabeling:

    def test_end_to_end_creates_outputs(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        out_csv = tmp_path / "labeled.csv"
        out_md = tmp_path / "summary.md"
        results = run_labeling(csv_path, out_csv, out_md)
        assert out_csv.exists()
        assert out_md.exists()

    def test_results_keys(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        results = run_labeling(
            csv_path,
            tmp_path / "labeled.csv",
            tmp_path / "summary.md",
        )
        for key in ("labeled_rows", "fieldnames", "label_counts", "rows_written"):
            assert key in results

    def test_rows_written_equals_study_count(self, tmp_path):
        csv_path = _make_fake_csv(tmp_path)
        results = run_labeling(
            csv_path,
            tmp_path / "labeled.csv",
            tmp_path / "summary.md",
        )
        assert results["rows_written"] == 6  # 6 studies in fixture


# ---------------------------------------------------------------------------
# Integration: real dataset (auto-skipped if not present)
# ---------------------------------------------------------------------------

def _find_real_csv() -> Path | None:
    candidates = [
        PROJECT_ROOT / "processed" / "iu_xray_index.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


REAL_CSV = _find_real_csv()


@pytest.mark.skipif(REAL_CSV is None, reason="Real iu_xray_index.csv not found.")
class TestRealDataset:

    def test_all_studies_labeled(self, tmp_path):
        labeled, _ = build_labeled_rows(REAL_CSV)
        assert len(labeled) == 2955, f"Expected 2955 studies, got {len(labeled)}"

    def test_labels_are_valid(self, tmp_path):
        labeled, _ = build_labeled_rows(REAL_CSV)
        for r in labeled:
            assert r["pneumonia_label"] in (POSITIVE, NEGATIVE, UNCERTAIN), \
                f"Invalid label: {r['pneumonia_label']}"

    def test_confidence_is_valid(self, tmp_path):
        labeled, _ = build_labeled_rows(REAL_CSV)
        for r in labeled:
            assert r["labeling_confidence"] in (HIGH, MEDIUM, LOW), \
                f"Invalid confidence: {r['labeling_confidence']}"

    def test_no_label_is_empty(self, tmp_path):
        labeled, _ = build_labeled_rows(REAL_CSV)
        empty = [r["report_id"] for r in labeled if not r.get("pneumonia_label")]
        assert empty == [], f"Studies with empty label: {empty[:5]}"

    def test_positive_count_is_reasonable(self, tmp_path):
        labeled, _ = build_labeled_rows(REAL_CSV)
        pos_count = sum(1 for r in labeled if r["pneumonia_label"] == POSITIVE)
        # From our analysis: ~10-20 clear positives out of 2955 in this dataset
        # The dataset is overwhelmingly negative (it's a general chest X-ray dataset)
        assert 1 <= pos_count <= 200, f"Positive count seems wrong: {pos_count}"

    def test_full_run_on_real_data(self, tmp_path):
        out_csv = tmp_path / "labeled.csv"
        out_md = tmp_path / "summary.md"
        results = run_labeling(REAL_CSV, out_csv, out_md)
        assert results["rows_written"] == 2955
        assert out_csv.exists()
        assert out_md.exists()
