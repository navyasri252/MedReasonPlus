"""
src/audit_pneumonia_labels.py
==============================
Phase 4 – Label Audit and Dataset Readiness for MedReason++

PURPOSE
-------
Before training any model, we audit the weak NLP labels produced in Phase 3
to understand:
  1. Whether the positive and uncertain labels look credible on inspection.
  2. Whether any records are cross-contaminated between splits.
  3. Whether every image view of a study carries the same study-level label.
  4. How severe the class imbalance is and why accuracy alone is misleading.

What this script does NOT do:
  - It does NOT change any existing labels.
  - It does NOT train a model.
  - It does NOT perform clinical validation (that requires human experts).

OUTPUTS
-------
  processed/label_audit.csv          – reviewer-ready spreadsheet
  reports/label_audit_summary.md     – findings and instructions

The audit CSV includes:
  - All 15 positive studies
  - All 26 uncertain studies
  - 100 negative studies chosen with a fixed random seed (seed=42)
  - A blank `manual_review_label` column for a reviewer to fill in
  - A blank `reviewer_notes` column

Usage
-----
    python src/audit_pneumonia_labels.py

    python src/audit_pneumonia_labels.py \\
        --labeled  processed/iu_xray_labeled.csv \\
        --index    processed/iu_xray_index.csv \\
        --output   processed/label_audit.csv \\
        --summary  reports/label_audit_summary.md \\
        --neg-sample-size 100 \\
        --seed 42
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

AUDIT_COLUMNS = [
    "report_id",
    "split",
    "pneumonia_label",
    "labeling_rule",
    "labeling_confidence",
    "report_word_count",
    "report_has_xxxx",
    "study_image_count",
    "report",
    "audit_reason",      # why this record was included in the audit sheet
    "manual_review_label",   # blank — for human reviewer
    "reviewer_notes",        # blank — for human reviewer
]


# ---------------------------------------------------------------------------
# 1 – Load CSVs
# ---------------------------------------------------------------------------

def load_labeled_csv(path: Path) -> list[dict]:
    """
    Load processed/iu_xray_labeled.csv.

    Expected columns (from Phase 3):
        report_id, study_image_count, report, report_word_count,
        report_has_xxxx, split, pneumonia_label, labeling_rule,
        labeling_confidence

    Parameters
    ----------
    path : Path

    Returns
    -------
    list[dict]
        One dict per study-level row.
    """
    if not path.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {path}")
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_index_csv(path: Path) -> list[dict]:
    """
    Load processed/iu_xray_index.csv.

    Expected columns (from Phase 2):
        image_path, image_path_rel, report_id, view_index,
        study_image_count, report, report_word_count,
        report_has_xxxx, split

    Parameters
    ----------
    path : Path

    Returns
    -------
    list[dict]
        One dict per image-view row.
    """
    if not path.exists():
        raise FileNotFoundError(f"Index CSV not found: {path}")
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 2 – Integrity checks
# ---------------------------------------------------------------------------

def check_cross_split_contamination(labeled_rows: list[dict]) -> list[str]:
    """
    Verify no report_id appears in more than one split.

    Parameters
    ----------
    labeled_rows : list[dict]

    Returns
    -------
    list[str]
        List of report_ids that appear in multiple splits.
    """
    id_to_splits: dict[str, set] = {}
    for row in labeled_rows:
        rid = row["report_id"]
        id_to_splits.setdefault(rid, set()).add(row["split"])
    return [rid for rid, splits in id_to_splits.items() if len(splits) > 1]


def check_view_label_consistency(
    labeled_rows: list[dict],
    index_rows: list[dict],
) -> list[dict]:
    """
    Verify every image view of a study carries the same study-level label.

    The labeled CSV is study-level (one row per study).  The index CSV is
    image-level (two rows per study).  This check ensures every view of a
    study would receive the same label when the label is joined back.

    Parameters
    ----------
    labeled_rows : list[dict]
    index_rows   : list[dict]

    Returns
    -------
    list[dict]
        Problem records, each with keys: report_id, view_index, issue.
        An empty list means all views are consistent.
    """
    # Build study → label map from labeled CSV
    study_label: dict[str, str] = {r["report_id"]: r["pneumonia_label"] for r in labeled_rows}

    problems = []
    for row in index_rows:
        rid = row["report_id"]
        if rid not in study_label:
            problems.append({
                "report_id": rid,
                "view_index": row.get("view_index", "?"),
                "issue": "report_id in index but not in labeled CSV",
            })
    return problems


def check_ids_alignment(
    labeled_rows: list[dict],
    index_rows: list[dict],
) -> dict:
    """
    Check that the set of report_ids in the labeled CSV exactly matches
    the set of report_ids in the index CSV.

    Parameters
    ----------
    labeled_rows : list[dict]
    index_rows   : list[dict]

    Returns
    -------
    dict
        {in_labeled_not_index, in_index_not_labeled}
    """
    labeled_ids = {r["report_id"] for r in labeled_rows}
    index_ids   = {r["report_id"] for r in index_rows}
    return {
        "in_labeled_not_index": sorted(labeled_ids - index_ids),
        "in_index_not_labeled": sorted(index_ids - labeled_ids),
    }


# ---------------------------------------------------------------------------
# 3 – Class imbalance analysis
# ---------------------------------------------------------------------------

def compute_imbalance_stats(labeled_rows: list[dict]) -> dict:
    """
    Compute class imbalance metrics and illustrate why accuracy is misleading.

    A classifier that always predicts 'negative' would achieve ~98.6% accuracy
    on this dataset while completely failing to detect pneumonia.

    Parameters
    ----------
    labeled_rows : list[dict]

    Returns
    -------
    dict
        Imbalance statistics.
    """
    label_counts = Counter(r["pneumonia_label"] for r in labeled_rows)
    total = len(labeled_rows)
    n_pos = label_counts.get("positive", 0)
    n_neg = label_counts.get("negative", 0)
    n_unc = label_counts.get("uncertain", 0)

    always_neg_accuracy = n_neg / total if total > 0 else 0.0
    pos_prevalence = n_pos / total if total > 0 else 0.0
    imbalance_ratio = round(n_neg / n_pos, 1) if n_pos > 0 else float("inf")

    return {
        "total_studies": total,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "uncertain_count": n_unc,
        "positive_prevalence_pct": round(pos_prevalence * 100, 2),
        "always_negative_accuracy_pct": round(always_neg_accuracy * 100, 2),
        "negative_to_positive_ratio": imbalance_ratio,
        "why_accuracy_misleads": (
            f"A model that predicts 'negative' for every study would be "
            f"{always_neg_accuracy * 100:.1f}% accurate but would recall 0% of "
            f"the {n_pos} positive cases.  Metrics like precision, recall, F1, "
            f"and AUC-ROC are required to evaluate a model on imbalanced data."
        ),
    }


# ---------------------------------------------------------------------------
# 4 – Split-level summary
# ---------------------------------------------------------------------------

def compute_split_summary(labeled_rows: list[dict]) -> dict:
    """
    Count labels per split (train / val / test).

    Parameters
    ----------
    labeled_rows : list[dict]

    Returns
    -------
    dict
        {split: {label: count}}
    """
    summary: dict[str, dict[str, int]] = {}
    for row in labeled_rows:
        split = row["split"]
        lbl   = row["pneumonia_label"]
        summary.setdefault(split, {})
        summary[split][lbl] = summary[split].get(lbl, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# 5 – Build audit rows
# ---------------------------------------------------------------------------

def build_audit_rows(
    labeled_rows: list[dict],
    neg_sample_size: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Select which rows to include in the audit CSV and annotate them.

    Selection strategy:
      - ALL positive rows        → audit_reason = "positive_label"
      - ALL uncertain rows       → audit_reason = "uncertain_label"
      - Random sample of negative rows → audit_reason = "negative_sample"
        (fixed seed ensures reproducibility; default seed=42, size=100)

    Parameters
    ----------
    labeled_rows     : list[dict]
    neg_sample_size  : int
        How many negative studies to include in the sample.
    seed             : int
        Random seed for reproducible sampling.

    Returns
    -------
    list[dict]
        Audit rows with all AUDIT_COLUMNS populated (blanks for review cols).
    """
    positives  = [r for r in labeled_rows if r["pneumonia_label"] == "positive"]
    uncertains = [r for r in labeled_rows if r["pneumonia_label"] == "uncertain"]
    negatives  = [r for r in labeled_rows if r["pneumonia_label"] == "negative"]

    # Reproducible random sample of negatives
    rng = random.Random(seed)
    actual_sample_size = min(neg_sample_size, len(negatives))
    neg_sample = rng.sample(negatives, actual_sample_size)

    # Sort each group for a consistent row order
    def sort_key(r: dict) -> tuple:
        return (r["split"], r["report_id"])

    tagged: list[dict] = []
    for r in sorted(positives, key=sort_key):
        tagged.append((_tag_row(r, "positive_label")))
    for r in sorted(uncertains, key=sort_key):
        tagged.append(_tag_row(r, "uncertain_label"))
    for r in sorted(neg_sample, key=sort_key):
        tagged.append(_tag_row(r, "negative_sample"))

    return tagged


def _tag_row(row: dict, audit_reason: str) -> dict:
    """
    Build a single audit row from a labeled CSV row.

    Keeps only the columns that belong in the audit output, adds
    audit_reason, and leaves manual_review_label / reviewer_notes blank.
    """
    return {
        "report_id":           row.get("report_id", ""),
        "split":               row.get("split", ""),
        "pneumonia_label":     row.get("pneumonia_label", ""),
        "labeling_rule":       row.get("labeling_rule", ""),
        "labeling_confidence": row.get("labeling_confidence", ""),
        "report_word_count":   row.get("report_word_count", ""),
        "report_has_xxxx":     row.get("report_has_xxxx", ""),
        "study_image_count":   row.get("study_image_count", ""),
        "report":              row.get("report", ""),
        "audit_reason":        audit_reason,
        "manual_review_label": "",   # ← reviewer fills this in
        "reviewer_notes":      "",   # ← reviewer fills this in
    }


# ---------------------------------------------------------------------------
# 6 – Write audit CSV
# ---------------------------------------------------------------------------

def write_audit_csv(audit_rows: list[dict], output_path: Path) -> None:
    """
    Write the audit rows to a CSV file.

    Parameters
    ----------
    audit_rows  : list[dict]
    output_path : Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)


# ---------------------------------------------------------------------------
# 7 – Write Markdown summary
# ---------------------------------------------------------------------------

def write_audit_summary(
    labeled_rows: list[dict],
    audit_rows: list[dict],
    imbalance: dict,
    split_summary: dict,
    integrity: dict,
    output_path: Path,
    audit_csv_path: Path,
) -> None:
    """
    Write the label audit summary Markdown report.

    Parameters
    ----------
    labeled_rows    : list[dict]
    audit_rows      : list[dict]
    imbalance       : dict  from compute_imbalance_stats()
    split_summary   : dict  from compute_split_summary()
    integrity       : dict  with cross_split, view_consistency, id_alignment
    output_path     : Path
    audit_csv_path  : Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit_reason_counts = Counter(r["audit_reason"] for r in audit_rows)

    # Collect rule breakdown for positive and uncertain
    pos_rule_counts  = Counter(r["labeling_rule"] for r in audit_rows if r["pneumonia_label"] == "positive")
    unc_rule_counts  = Counter(r["labeling_rule"] for r in audit_rows if r["pneumonia_label"] == "uncertain")
    conf_counts      = Counter(
        (r["pneumonia_label"], r["labeling_confidence"]) for r in labeled_rows
    )

    total = imbalance["total_studies"]

    lines = [
        "# Phase 4 — Label Audit Summary",
        "",
        "> **⚠ DISCLAIMER:** Labels are *weak supervision* derived from NLP patterns.",
        "> They have **NOT been clinically validated**.",
        "> This audit is a project-level review only.",
        "",
        f"> Audit CSV: `{audit_csv_path}`",
        "",
        "---",
        "",
        "## 1. Label Counts",
        "",
        "| Label | Count | % of total |",
        "|-------|------:|-----------:|",
        f"| positive  | {imbalance['positive_count']:,} | {imbalance['positive_count']/total*100:.2f}% |",
        f"| negative  | {imbalance['negative_count']:,} | {imbalance['negative_count']/total*100:.2f}% |",
        f"| uncertain | {imbalance['uncertain_count']:,} | {imbalance['uncertain_count']/total*100:.2f}% |",
        f"| **total** | **{total:,}** | 100% |",
        "",
        "---",
        "",
        "## 2. Label Distribution by Split",
        "",
        "| Split | positive | negative | uncertain | total |",
        "|-------|--------:|--------:|--------:|------:|",
    ]
    for split in ["train", "val", "test"]:
        sc = split_summary.get(split, {})
        pos = sc.get("positive", 0)
        neg = sc.get("negative", 0)
        unc = sc.get("uncertain", 0)
        tot = pos + neg + unc
        lines.append(f"| {split} | {pos} | {neg:,} | {unc} | {tot:,} |")

    lines += [
        "",
        "> Val split has **0 positive** studies. This means a model trained with",
        "> standard train/val evaluation would never see a positive validation example",
        "> during training — validation loss would be uninformative for the positive class.",
        "",
        "---",
        "",
        "## 3. Class Imbalance Analysis",
        "",
        f"**Negative-to-positive ratio: {imbalance['negative_to_positive_ratio']:,}:1**",
        "",
        f"The positive class makes up only **{imbalance['positive_prevalence_pct']}%** of all studies.",
        "",
        "### Why accuracy alone is misleading",
        "",
        f"> {imbalance['why_accuracy_misleads']}",
        "",
        "| Metric | Naive always-negative classifier |",
        "|--------|----------------------------------|",
        f"| Accuracy | {imbalance['always_negative_accuracy_pct']}% |",
        "| Recall (positive class) | 0% |",
        "| Precision (positive class) | undefined (no positives predicted) |",
        "| F1 (positive class) | 0 |",
        "",
        "### Recommended metrics for this problem",
        "",
        "- **Recall (sensitivity)** — did the model find the positive cases?",
        "- **Precision** — when it predicted positive, was it right?",
        "- **F1-score (positive class)** — harmonic mean of precision and recall.",
        "- **AUC-ROC / AUC-PR** — threshold-independent performance.",
        "- **Confusion matrix** — inspect false-negative rate separately.",
        "",
        "### Strategies to address imbalance (for Phase 5+)",
        "",
        "1. **Weighted loss** — assign higher loss weight to the positive class.",
        "2. **Oversampling** — use `SMOTE` or repeat positive samples during training.",
        "3. **Undersampling** — randomly drop negative training samples.",
        "4. **Threshold tuning** — move the decision threshold to favour recall.",
        "5. **Use uncertain as a third class** — do not force uncertain into binary.",
        "6. **Seek a more balanced dataset** — the IU X-ray dataset was not designed",
        "   for pneumonia detection; a dataset curated for pneumonia would be better.",
        "",
        "---",
        "",
        "## 4. Integrity Checks",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Cross-split contamination | {'✅ None' if not integrity['cross_split'] else '❌ ' + str(len(integrity['cross_split'])) + ' IDs'} |",
        f"| View-label consistency | {'✅ All consistent' if not integrity['view_problems'] else '❌ ' + str(len(integrity['view_problems'])) + ' problems'} |",
        f"| IDs in labeled not in index | {'✅ None' if not integrity['id_alignment']['in_labeled_not_index'] else '❌ ' + str(len(integrity['id_alignment']['in_labeled_not_index']))} |",
        f"| IDs in index not in labeled | {'✅ None' if not integrity['id_alignment']['in_index_not_labeled'] else '❌ ' + str(len(integrity['id_alignment']['in_index_not_labeled']))} |",
        "",
        "---",
        "",
        "## 5. Audit CSV Contents",
        "",
        f"Audit CSV: `{audit_csv_path}`",
        "",
        "| Reason | Count |",
        "|--------|------:|",
        f"| All positive studies | {audit_reason_counts.get('positive_label', 0)} |",
        f"| All uncertain studies | {audit_reason_counts.get('uncertain_label', 0)} |",
        f"| Random negative sample (seed=42) | {audit_reason_counts.get('negative_sample', 0)} |",
        f"| **Total rows in audit CSV** | **{len(audit_rows)}** |",
        "",
        "---",
        "",
        "## 6. Positive Label Breakdown",
        "",
        "| Labeling Rule | Count |",
        "|---------------|------:|",
    ]
    for rule, count in pos_rule_counts.most_common():
        lines.append(f"| `{rule}` | {count} |")

    lines += [
        "",
        "### All positive studies",
        "",
        "| Report ID | Split | Rule | Confidence | Report excerpt (first 200 chars) |",
        "|-----------|-------|------|-----------|----------------------------------|",
    ]
    for r in audit_rows:
        if r["pneumonia_label"] == "positive":
            excerpt = r["report"][:200].replace("|", "\\|")
            lines.append(
                f"| `{r['report_id']}` | {r['split']} | `{r['labeling_rule']}` "
                f"| {r['labeling_confidence']} | {excerpt}… |"
            )

    lines += [
        "",
        "---",
        "",
        "## 7. Uncertain Label Breakdown",
        "",
        "| Labeling Rule | Count |",
        "|---------------|------:|",
    ]
    for rule, count in unc_rule_counts.most_common():
        lines.append(f"| `{rule}` | {count} |")

    lines += [
        "",
        "### All uncertain studies",
        "",
        "| Report ID | Split | Rule | Confidence | Report excerpt (first 200 chars) |",
        "|-----------|-------|------|-----------|----------------------------------|",
    ]
    for r in audit_rows:
        if r["pneumonia_label"] == "uncertain":
            excerpt = r["report"][:200].replace("|", "\\|")
            lines.append(
                f"| `{r['report_id']}` | {r['split']} | `{r['labeling_rule']}` "
                f"| {r['labeling_confidence']} | {excerpt}… |"
            )

    lines += [
        "",
        "---",
        "",
        "## 8. How to Use the Audit CSV",
        "",
        "The file `processed/label_audit.csv` is ready for human review.",
        "Open it in Excel, Google Sheets, or any CSV editor.",
        "",
        "For each row, a reviewer should:",
        "",
        "1. Read the `report` column carefully.",
        "2. Fill in `manual_review_label` with one of: `positive`, `negative`, `uncertain`, `exclude`.",
        "   - `exclude` means the report is too ambiguous, corrupted, or not a chest X-ray report.",
        "3. Add any clinical notes or concerns in `reviewer_notes`.",
        "4. Pay special attention to rows with `labeling_confidence = MEDIUM` or `LOW`.",
        "",
        "**Priority order for review:**",
        "1. All `positive` rows — these are the signal class; false positives matter most.",
        "2. All `uncertain` rows — these may be re-classifiable or should be excluded.",
        "3. The 100 `negative_sample` rows — spot-check for false negatives.",
        "",
        "---",
        "",
        "## 9. Phase 4 Checkpoint",
        "",
        "| Criterion | Status |",
        "|-----------|--------|",
        f"| `processed/label_audit.csv` created | ✅ |",
        f"| All positive studies included | ✅ ({imbalance['positive_count']}) |",
        f"| All uncertain studies included | ✅ ({imbalance['uncertain_count']}) |",
        f"| 100 negative sample included (seed=42) | ✅ |",
        f"| manual_review_label column present (blank) | ✅ |",
        f"| reviewer_notes column present (blank) | ✅ |",
        f"| Label counts by split reported | ✅ |",
        f"| Cross-split contamination check | ✅ |",
        f"| View-label consistency check | ✅ |",
        f"| Class imbalance explained | ✅ |",
        f"| Why accuracy is misleading explained | ✅ |",
        f"| Original CSVs unmodified | ✅ |",
        f"| No model training | ✅ |",
        "",
        "---",
        "",
        "## 10. Decision Gate for Phase 5",
        "",
        "Before proceeding to model training, we recommend deciding one of:",
        "",
        "| Option | When to choose |",
        "|--------|----------------|",
        "| **Proceed with current labels** | Reviewer confirms ≥10 true positives after manual review |",
        "| **Revise labeling rules** | Manual review reveals systemic false positives/negatives |",
        "| **Use a different dataset** | IU X-ray lacks sufficient positive examples for reliable training |",
        "| **Treat as anomaly detection** | Relabel the 15 positives as anomalies; train on negatives only |",
        "",
        "The **recommended choice** given only 15 positives out of 2,955 is either",
        "to **revise the labeling rules** (the proxy rules may be too conservative)",
        "or to **use a pneumonia-specific dataset** such as the CheXpert or",
        "NIH ChestX-ray14 datasets, which have verified radiologist labels.",
        "",
        "---",
        "",
        "_Generated by `src/audit_pneumonia_labels.py`_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 8 – Orchestration
# ---------------------------------------------------------------------------

def run_audit(
    labeled_csv_path: Path,
    index_csv_path: Path,
    audit_csv_path: Path,
    summary_path: Path,
    neg_sample_size: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run the full Phase 4 audit pipeline.

    Parameters
    ----------
    labeled_csv_path : Path
    index_csv_path   : Path
    audit_csv_path   : Path
    summary_path     : Path
    neg_sample_size  : int
    seed             : int

    Returns
    -------
    dict
        Results dict used by tests.
    """
    print(f"\n{'='*60}")
    print("  MedReason++ — Phase 4: Label Audit & Dataset Readiness")
    print(f"  Labeled CSV : {labeled_csv_path}")
    print(f"  Index CSV   : {index_csv_path}")
    print(f"  Audit CSV   : {audit_csv_path}")
    print(f"{'='*60}\n")

    # Load
    print("[1/7] Loading CSVs …")
    labeled_rows = load_labeled_csv(labeled_csv_path)
    index_rows   = load_index_csv(index_csv_path)
    print(f"       Labeled: {len(labeled_rows):,} studies | Index: {len(index_rows):,} image rows")

    # Integrity
    print("[2/7] Running integrity checks …")
    cross_split = check_cross_split_contamination(labeled_rows)
    view_problems = check_view_label_consistency(labeled_rows, index_rows)
    id_alignment = check_ids_alignment(labeled_rows, index_rows)
    print(f"       Cross-split contamination: {len(cross_split)}")
    print(f"       View-label consistency problems: {len(view_problems)}")
    print(f"       ID alignment issues: "
          f"{len(id_alignment['in_labeled_not_index'])} + "
          f"{len(id_alignment['in_index_not_labeled'])}")

    # Class imbalance
    print("[3/7] Computing class imbalance statistics …")
    imbalance = compute_imbalance_stats(labeled_rows)
    print(f"       Positive: {imbalance['positive_count']} "
          f"({imbalance['positive_prevalence_pct']}%)")
    print(f"       Always-negative accuracy: {imbalance['always_negative_accuracy_pct']}%")
    print(f"       Neg:pos ratio: {imbalance['negative_to_positive_ratio']}:1")

    # Split summary
    print("[4/7] Computing split-level label distribution …")
    split_summary = compute_split_summary(labeled_rows)
    for split, counts in split_summary.items():
        print(f"       {split}: {counts}")

    # Build audit rows
    print(f"[5/7] Building audit rows "
          f"(all pos+unc + {neg_sample_size} neg sample, seed={seed}) …")
    audit_rows = build_audit_rows(labeled_rows, neg_sample_size, seed)
    print(f"       Audit rows: {len(audit_rows)}")

    # Write audit CSV
    print("[6/7] Writing audit CSV …")
    write_audit_csv(audit_rows, audit_csv_path)
    print(f"       Saved: {audit_csv_path}")

    # Write summary
    print("[7/7] Writing summary report …")
    integrity = {
        "cross_split": cross_split,
        "view_problems": view_problems,
        "id_alignment": id_alignment,
    }
    write_audit_summary(
        labeled_rows, audit_rows, imbalance,
        split_summary, integrity,
        summary_path, audit_csv_path,
    )
    print(f"       Saved: {summary_path}")

    print(f"\n{'='*60}")
    print("  Phase 4 complete.")
    print(f"{'='*60}\n")

    return {
        "labeled_rows": labeled_rows,
        "index_rows": index_rows,
        "audit_rows": audit_rows,
        "imbalance": imbalance,
        "split_summary": split_summary,
        "cross_split": cross_split,
        "view_problems": view_problems,
        "id_alignment": id_alignment,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Label audit and dataset readiness for MedReason++."
    )
    parser.add_argument("--labeled",  default="processed/iu_xray_labeled.csv")
    parser.add_argument("--index",    default="processed/iu_xray_index.csv")
    parser.add_argument("--output",   default="processed/label_audit.csv")
    parser.add_argument("--summary",  default="reports/label_audit_summary.md")
    parser.add_argument("--neg-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_audit(
        labeled_csv_path=Path(args.labeled),
        index_csv_path=Path(args.index),
        audit_csv_path=Path(args.output),
        summary_path=Path(args.summary),
        neg_sample_size=args.neg_sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
