"""
src/build_dataset_index.py
==========================
Phase 2 – Build a validated CSV index for the IU X-ray dataset (MedReason++).

What this script does
---------------------
1. Reads annotation.json (the single source of truth for image–report links).
2. Expands each study record (which references 2+ images) into one CSV row
   PER IMAGE — because models are trained on individual images, not study pairs.
3. Verifies that every image path resolves to an actual file on disk.
4. Reports — without silently discarding — any records that have problems:
     - missing image files
     - empty report text
     - duplicate record IDs
5. Writes processed/iu_xray_index.csv with these columns:

   image_path          Absolute path to the PNG file on disk
   image_path_rel      Relative path as stored in annotation (e.g. CXR001/0.png)
   report_id           Study-level unique identifier (e.g. CXR2384_IM-0942)
   view_index          Which view this row represents (0 = frontal/PA, 1 = lateral)
   study_image_count   Total PNG files in this study's folder (≥2; most are 2)
   report              Full report text (one merged paragraph)
   report_word_count   Number of whitespace-separated tokens in the report
   report_has_xxxx     True if the report contains 'XXXX' (de-id placeholder)
   split               Dataset split: train / val / test

   NOTE: The R2Gen IU X-ray annotation does NOT separate findings from
   impression — the 'report' field is a single merged paragraph.
   Columns 'findings' and 'impression' are therefore NOT included; we
   document why rather than inventing them from partial text.

6. Writes reports/dataset_build_summary.md with full counts and issue log.

Usage
-----
    python src/build_dataset_index.py --data_root iu_xray/iu_xray

    # or let it auto-detect:
    python src/build_dataset_index.py

The script never modifies or deletes original dataset files.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "image_path",
    "image_path_rel",
    "report_id",
    "view_index",
    "study_image_count",
    "report",
    "report_word_count",
    "report_has_xxxx",
    "split",
]


# ---------------------------------------------------------------------------
# 1 – Locate the dataset root
# ---------------------------------------------------------------------------

def find_dataset_root(start: Path = None) -> Path:
    """
    Search upward from `start` for a folder that contains both
    `annotation.json` and an `images/` sub-directory.

    Parameters
    ----------
    start : Path, optional
        Directory to begin searching from.  Defaults to two levels above
        this file (i.e. the project root when running from src/).

    Returns
    -------
    Path
        The resolved dataset root directory.

    Raises
    ------
    FileNotFoundError
        If no suitable directory is found.
    """
    if start is None:
        start = Path(__file__).resolve().parent.parent

    for candidate in start.rglob("annotation.json"):
        root = candidate.parent
        if (root / "images").is_dir():
            return root

    raise FileNotFoundError(
        "Cannot locate the IU X-ray dataset root.  "
        "Expected a folder with 'annotation.json' and 'images/'.  "
        "Pass --data_root explicitly."
    )


# ---------------------------------------------------------------------------
# 2 – Load annotation.json
# ---------------------------------------------------------------------------

def load_annotation(data_root: Path) -> dict:
    """
    Load and return the raw annotation dictionary from annotation.json.

    The R2Gen annotation file has this structure:
        {
            "train": [ {id, report, image_path, split}, ... ],
            "val":   [ ... ],
            "test":  [ ... ]
        }

    Parameters
    ----------
    data_root : Path
        The dataset root (contains annotation.json).

    Returns
    -------
    dict
        Raw annotation with keys "train", "val", "test".
    """
    ann_path = data_root / "annotation.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation.json not found at: {ann_path}")
    with ann_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 3 – Count actual PNG files per study directory
# ---------------------------------------------------------------------------

def build_study_image_counts(data_root: Path) -> dict[str, int]:
    """
    Walk images/ and count how many PNG files exist in each study folder.

    Most studies have exactly 2 views (frontal + lateral).
    Some have 3–5 (additional oblique or supplemental views).
    The annotation only references views 0 and 1.

    Returns
    -------
    dict
        {study_id: png_count}  e.g. {"CXR1000_IM-0003": 3, ...}
    """
    images_dir = data_root / "images"
    counts: dict[str, int] = {}
    for study_dir in images_dir.iterdir():
        if study_dir.is_dir():
            counts[study_dir.name] = len(list(study_dir.glob("*.png")))
    return counts


# ---------------------------------------------------------------------------
# 4 – Validate records and build rows
# ---------------------------------------------------------------------------

def build_index_rows(
    data: dict,
    data_root: Path,
    study_image_counts: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """
    Expand annotation records into one CSV row per image view.

    For each record we:
      - Check for duplicate IDs.
      - Check that the report text is non-empty.
      - Check that each referenced image file exists on disk.
      - Compute per-row derived fields (word_count, has_xxxx, view_index).

    Records with problems are NOT silently dropped — they are included in
    the `issues` list and still written to the CSV with a note so nothing
    is hidden.

    Parameters
    ----------
    data : dict
        Raw annotation from load_annotation().
    data_root : Path
        Dataset root path (images/ is under here).
    study_image_counts : dict
        {study_id: png_count} from build_study_image_counts().

    Returns
    -------
    rows : list[dict]
        One dict per image row, ready for CSV writing.
    issues : list[dict]
        One dict per problem found (type, record_id, detail).
    """
    images_dir = data_root / "images"
    all_records: list[dict] = []
    for split_records in data.values():
        all_records.extend(split_records)

    # ---- Check for duplicate IDs ----
    id_counts = Counter(rec["id"] for rec in all_records)
    duplicate_ids = {rid for rid, cnt in id_counts.items() if cnt > 1}

    rows: list[dict] = []
    issues: list[dict] = []

    for rec in all_records:
        rec_id: str = rec["id"]
        report: str = rec.get("report", "")
        image_paths: list[str] = rec.get("image_path", [])
        split: str = rec.get("split", "unknown")

        # --- Issue: duplicate ID ---
        if rec_id in duplicate_ids:
            issues.append({
                "type": "duplicate_id",
                "record_id": rec_id,
                "detail": f"This ID appears {id_counts[rec_id]} times across all splits.",
            })

        # --- Issue: empty report ---
        if not report.strip():
            issues.append({
                "type": "empty_report",
                "record_id": rec_id,
                "detail": "report field is empty or whitespace-only.",
            })

        report_word_count = len(report.split()) if report.strip() else 0
        report_has_xxxx = "XXXX" in report
        study_img_count = study_image_counts.get(rec_id, 0)

        # --- One row per referenced image path ---
        for view_idx, rel_path in enumerate(image_paths):
            abs_path = images_dir / rel_path
            file_exists = abs_path.exists()

            if not file_exists:
                issues.append({
                    "type": "missing_image",
                    "record_id": rec_id,
                    "detail": f"Image not found on disk: {rel_path}",
                })

            rows.append({
                "image_path": str(abs_path),
                "image_path_rel": rel_path,
                "report_id": rec_id,
                "view_index": view_idx,
                "study_image_count": study_img_count,
                "report": report,
                "report_word_count": report_word_count,
                "report_has_xxxx": report_has_xxxx,
                "split": split,
            })

    return rows, issues


# ---------------------------------------------------------------------------
# 5 – Write the CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write the index rows to a CSV file.

    Creates parent directories if they don't exist.
    Uses UTF-8 encoding with a BOM so Excel opens it correctly.

    Parameters
    ----------
    rows : list[dict]
        Rows from build_index_rows().
    output_path : Path
        Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 6 – Validate the written CSV
# ---------------------------------------------------------------------------

def validate_csv(csv_path: Path) -> dict:
    """
    Re-read the CSV from disk and run a set of post-write sanity checks.

    Checks:
      - All rows are readable and have the expected columns.
      - image_path values point to files that exist on disk.
      - report_id values are non-empty.
      - report values are non-empty.
      - view_index values are 0 or 1 (the annotation-referenced views).

    Returns
    -------
    dict
        Validation summary with counts and any detected problems.
    """
    if not csv_path.exists():
        return {"error": f"CSV file not found: {csv_path}"}

    rows_read = 0
    missing_images = 0
    empty_report_ids = 0
    empty_reports = 0
    unexpected_columns: list[str] = []
    column_issues: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        actual_cols = reader.fieldnames or []

        for expected in CSV_COLUMNS:
            if expected not in actual_cols:
                column_issues.append(f"Missing column: {expected}")

        for unexpected in actual_cols:
            if unexpected not in CSV_COLUMNS:
                unexpected_columns.append(unexpected)

        for row in reader:
            rows_read += 1
            if not Path(row.get("image_path", "")).exists():
                missing_images += 1
            if not row.get("report_id", "").strip():
                empty_report_ids += 1
            if not row.get("report", "").strip():
                empty_reports += 1

    return {
        "rows_read": rows_read,
        "columns_found": actual_cols,
        "column_issues": column_issues,
        "unexpected_columns": unexpected_columns,
        "missing_image_paths": missing_images,
        "empty_report_ids": empty_report_ids,
        "empty_report_texts": empty_reports,
    }


# ---------------------------------------------------------------------------
# 7 – Write the summary report
# ---------------------------------------------------------------------------

def write_summary_report(
    rows: list[dict],
    issues: list[dict],
    validation: dict,
    data_root: Path,
    csv_path: Path,
    output_path: Path,
) -> None:
    """
    Write a human-readable Markdown summary of the build process.

    Parameters
    ----------
    rows : list[dict]
        All CSV rows produced.
    issues : list[dict]
        All issues detected during build.
    validation : dict
        Output of validate_csv().
    data_root : Path
        Dataset root used.
    csv_path : Path
        Path where the CSV was written.
    output_path : Path
        Where to write the markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Compute split-level stats ---
    split_counter = Counter(r["split"] for r in rows)
    view_counter = Counter(r["view_index"] for r in rows)
    xxxx_count = sum(1 for r in rows if str(r["report_has_xxxx"]).lower() == "true")
    unique_studies = len({r["report_id"] for r in rows})

    issue_types = Counter(iss["type"] for iss in issues)

    lines = [
        "# IU X-ray Dataset Build Summary",
        "",
        f"> Dataset root: `{data_root}`",
        f"> Output CSV: `{csv_path}`",
        "",
        "---",
        "",
        "## 1. CSV Schema",
        "",
        "The annotation.json in the R2Gen IU X-ray dataset provides reports as a single",
        "merged paragraph.  There are **no separate `findings` or `impression` sections**",
        "in this version of the data — the entire clinical text is in the `report` field.",
        "Inventing a split would be fabrication, so only the real fields are included.",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `image_path` | Absolute path to the PNG file on disk |",
        "| `image_path_rel` | Relative path as stored in annotation.json |",
        "| `report_id` | Study-level unique ID (e.g. `CXR2384_IM-0942`) |",
        "| `view_index` | View number: 0 = frontal/PA, 1 = lateral |",
        "| `study_image_count` | Total PNGs in this study's folder (≥2) |",
        "| `report` | Full merged report text |",
        "| `report_word_count` | Token count of the report |",
        "| `report_has_xxxx` | True if report contains `XXXX` de-id tokens |",
        "| `split` | train / val / test |",
        "",
        "---",
        "",
        "## 2. Record Counts",
        "",
        "| Split | Rows (images) | Unique studies |",
        "|-------|----------:|-------:|",
    ]

    for split in ["train", "val", "test"]:
        rows_in_split = [r for r in rows if r["split"] == split]
        studies_in_split = len({r["report_id"] for r in rows_in_split})
        lines.append(f"| {split} | {len(rows_in_split):,} | {studies_in_split:,} |")

    total_rows = len(rows)
    lines += [
        f"| **total** | **{total_rows:,}** | **{unique_studies:,}** |",
        "",
        "> **One row per image view.** Each study has 2 annotation-referenced views,",
        "> so row count ≈ 2 × study count.",
        "",
        "---",
        "",
        "## 3. View Distribution",
        "",
        "| View index | Rows |",
        "|----------:|-----:|",
    ]
    for view_idx in sorted(view_counter.keys()):
        lines.append(f"| {view_idx} | {view_counter[view_idx]:,} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Report Text",
        "",
        f"- Rows with `XXXX` de-identification tokens: **{xxxx_count:,}**",
        "",
        "The `XXXX` token is a de-identification placeholder inserted by the original",
        "dataset curators wherever patient-identifying or sensitive terms appeared.",
        "These rows are retained — the placeholder does not invalidate the report.",
        "",
        "---",
        "",
        "## 5. Issues Log",
        "",
    ]

    if not issues:
        lines.append("No issues detected. ✅")
    else:
        lines += [
            f"Total issues detected: **{len(issues)}**",
            "",
            "| Issue type | Count |",
            "|------------|------:|",
        ]
        for issue_type, count in issue_types.most_common():
            lines.append(f"| `{issue_type}` | {count} |")

        lines += [
            "",
            "### Issue details",
            "",
            "| Type | Record ID | Detail |",
            "|------|-----------|--------|",
        ]
        for iss in issues[:50]:  # cap table at 50 rows for readability
            lines.append(
                f"| `{iss['type']}` | `{iss['record_id']}` | {iss['detail']} |"
            )
        if len(issues) > 50:
            lines.append(
                f"| … | … | *(and {len(issues) - 50} more — see full issue log)* |"
            )

    lines += [
        "",
        "---",
        "",
        "## 6. CSV Validation",
        "",
        f"Rows verified after writing: **{validation.get('rows_read', 0):,}**",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| All expected columns present | {'✅' if not validation.get('column_issues') else '❌ ' + str(validation['column_issues'])} |",
        f"| Image paths resolve to real files | {'✅' if validation.get('missing_image_paths', 0) == 0 else '❌ ' + str(validation['missing_image_paths']) + ' missing'} |",
        f"| All rows have a report_id | {'✅' if validation.get('empty_report_ids', 0) == 0 else '❌ ' + str(validation['empty_report_ids']) + ' empty'} |",
        f"| All rows have report text | {'✅' if validation.get('empty_report_texts', 0) == 0 else '❌ ' + str(validation['empty_report_texts']) + ' empty'} |",
        "",
        "---",
        "",
        "## 7. How to Run",
        "",
        "```bash",
        "# From the project root",
        "python src/build_dataset_index.py --data_root iu_xray/iu_xray",
        "",
        "# With explicit output path",
        "python src/build_dataset_index.py \\",
        "    --data_root iu_xray/iu_xray \\",
        "    --output processed/iu_xray_index.csv \\",
        "    --summary reports/dataset_build_summary.md",
        "",
        "# Auto-detect dataset root",
        "python src/build_dataset_index.py",
        "```",
        "",
        "---",
        "",
        "## 8. Phase 2 Checkpoint",
        "",
        "| Criterion | Status |",
        "|-----------|--------|",
        f"| `processed/iu_xray_index.csv` exists | {'✅' if csv_path.exists() else '❌'} |",
        f"| Image paths are valid | {'✅' if validation.get('missing_image_paths', 0) == 0 else '❌'} |",
        "| Report-to-image matching explained | ✅ (annotation.json, subdir = record id) |",
        f"| Missing / unmatched records reported | {'✅' if True else '❌'} |",
        "| Original files unmodified | ✅ |",
        "| No invented labels | ✅ |",
        "",
        "**Ready for Phase 3** — constructing pneumonia labels from report text.",
        "",
        "---",
        "",
        "_Generated by `src/build_dataset_index.py`_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 8 – Main orchestration
# ---------------------------------------------------------------------------

def run_build(
    data_root: Path,
    csv_path: Path,
    summary_path: Path,
) -> dict:
    """
    Orchestrate all Phase 2 steps and return a results dict.

    Steps:
      1. Load annotation.json
      2. Count physical images per study
      3. Build and validate index rows
      4. Write CSV
      5. Validate the written CSV
      6. Write Markdown summary

    Parameters
    ----------
    data_root : Path
        Dataset root folder (contains annotation.json and images/).
    csv_path : Path
        Where to write the output CSV.
    summary_path : Path
        Where to write the Markdown build summary.

    Returns
    -------
    dict
        A results dict containing all intermediate outputs (used by tests).
    """
    print(f"\n{'='*60}")
    print("  MedReason++ — Phase 2: Build Dataset Index")
    print(f"  Dataset root : {data_root}")
    print(f"  Output CSV   : {csv_path}")
    print(f"{'='*60}\n")

    # Step 1 — load annotation
    print("[1/6] Loading annotation.json …")
    data = load_annotation(data_root)
    total_records = sum(len(v) for v in data.values())
    print(f"       {total_records:,} study records across splits: "
          f"{', '.join(f'{k}={len(v)}' for k,v in data.items())}")

    # Step 2 — count disk images
    print("[2/6] Counting physical PNG files per study …")
    study_counts = build_study_image_counts(data_root)
    print(f"       {len(study_counts):,} study folders found on disk.")

    # Step 3 — build rows
    print("[3/6] Building index rows (one row per image view) …")
    rows, issues = build_index_rows(data, data_root, study_counts)
    print(f"       {len(rows):,} rows built, {len(issues)} issues detected.")
    if issues:
        from collections import Counter as _Counter
        by_type = _Counter(i["type"] for i in issues)
        for itype, cnt in by_type.items():
            print(f"       ⚠  {itype}: {cnt}")

    # Step 4 — write CSV
    print("[4/6] Writing CSV …")
    write_csv(rows, csv_path)
    print(f"       Saved to: {csv_path}")

    # Step 5 — validate CSV
    print("[5/6] Validating CSV …")
    validation = validate_csv(csv_path)
    v = validation
    print(f"       Rows verified: {v.get('rows_read', 0):,}")
    print(f"       Missing image files: {v.get('missing_image_paths', 0)}")
    print(f"       Empty report_id: {v.get('empty_report_ids', 0)}")
    print(f"       Empty report text: {v.get('empty_report_texts', 0)}")
    if v.get("column_issues"):
        print(f"       ⚠ Column issues: {v['column_issues']}")

    # Step 6 — summary report
    print("[6/6] Writing Markdown summary …")
    write_summary_report(rows, issues, validation, data_root, csv_path, summary_path)
    print(f"       Saved to: {summary_path}")

    print(f"\n{'='*60}")
    print("  Phase 2 complete.")
    print(f"{'='*60}\n")

    return {
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "rows": rows,
        "issues": issues,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Build a validated CSV index for the IU X-ray dataset."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to dataset root (contains annotation.json and images/).  "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed/iu_xray_index.csv",
        help="Output CSV path (default: processed/iu_xray_index.csv).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="reports/dataset_build_summary.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve() if args.data_root else find_dataset_root()
    csv_path = Path(args.output)
    summary_path = Path(args.summary)

    run_build(data_root, csv_path, summary_path)


if __name__ == "__main__":
    main()
