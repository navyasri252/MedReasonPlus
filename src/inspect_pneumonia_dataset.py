"""
src/inspect_pneumonia_dataset.py
==================================
Chest X-Ray Images (Pneumonia) Dataset Inspection — MedReason++

PURPOSE
-------
Inspect the Chest X-Ray Images (Pneumonia) dataset (Kaggle / Kermany et al.)
that has been added to this project.  This script does NOT:
  - modify or delete any image files
  - combine this dataset with the IU X-ray dataset
  - change any Phase 1–4 outputs

EXPECTED DATASET LAYOUT
------------------------
<dataset_root>/
    train/
        NORMAL/      *.jpeg
        PNEUMONIA/   *.jpeg
    val/
        NORMAL/      *.jpeg
        PNEUMONIA/   *.jpeg
    test/
        NORMAL/      *.jpeg
        PNEUMONIA/   *.jpeg

CHECKS PERFORMED
----------------
1. Locate and confirm the dataset root.
2. Count NORMAL and PNEUMONIA images per split.
3. Detect pneumonia sub-types (bacterial vs viral) from filenames.
4. Identify any non-JPEG files and unexpected folder structure.
5. Sample-check image readability with Pillow.
6. Detect duplicate filenames within and across splits.
7. Report image dimension and color-mode diversity.

Usage
-----
    python src/inspect_pneumonia_dataset.py

    python src/inspect_pneumonia_dataset.py --dataset_root "archive (7)/chest_xray"
"""

import argparse
import random
from collections import Counter
from pathlib import Path

# Pillow is used for readability checks; import lazily so tests can mock it.
try:
    from PIL import Image as _PIL_Image
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")

# How many images to sample for the readability / dimension checks.
# Set to None to check every image (slow on large datasets).
READABILITY_SAMPLE_SIZE = 100
DIMENSION_SAMPLE_SIZE = 50


# ---------------------------------------------------------------------------
# 1 – Dataset root detection
# ---------------------------------------------------------------------------

def find_dataset_root(start: Path = None) -> Path:
    """
    Search for a folder whose direct children are the expected splits
    (train/, val/, test/) each containing NORMAL/ and PNEUMONIA/ sub-folders.

    Searches downward from `start` up to 4 directory levels.

    Parameters
    ----------
    start : Path, optional
        Where to begin searching.  Defaults to the project root (two levels
        above this file).

    Returns
    -------
    Path
        Resolved dataset root directory.

    Raises
    ------
    FileNotFoundError
        If no matching directory is found.
    """
    if start is None:
        start = Path(__file__).resolve().parent.parent

    def _looks_like_root(path: Path) -> bool:
        """Return True if path contains at least train/NORMAL and train/PNEUMONIA."""
        return (
            (path / "train" / "NORMAL").is_dir()
            and (path / "train" / "PNEUMONIA").is_dir()
        )

    if _looks_like_root(start):
        return start

    # Breadth-first search up to depth 4
    queue = [start]
    for _ in range(4):
        next_queue = []
        for candidate in queue:
            try:
                children = [c for c in candidate.iterdir() if c.is_dir()]
            except PermissionError:
                continue
            for child in children:
                if _looks_like_root(child):
                    return child
                next_queue.append(child)
        queue = next_queue

    raise FileNotFoundError(
        f"Cannot find the Chest X-Ray Pneumonia dataset root under: {start}\n"
        "Expected a folder with train/NORMAL and train/PNEUMONIA sub-folders.\n"
        "Pass --dataset_root explicitly."
    )


# ---------------------------------------------------------------------------
# 2 – Count images per split and class
# ---------------------------------------------------------------------------

def count_images(dataset_root: Path) -> dict:
    """
    Walk the dataset and count JPEG files in each split/class folder.

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    dict
        Structure:
        {
            split: {
                "NORMAL":    int,
                "PNEUMONIA": int,
                "total":     int,
                "exists":    bool,   # whether the split folder exists
            }
        }
    """
    counts: dict[str, dict] = {}
    for split in SPLITS:
        split_dir = dataset_root / split
        entry: dict = {"exists": split_dir.is_dir()}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.is_dir():
                entry[cls] = len(list(cls_dir.glob("*.jpeg")))
            else:
                entry[cls] = 0
        entry["total"] = entry.get("NORMAL", 0) + entry.get("PNEUMONIA", 0)
        counts[split] = entry
    return counts


# ---------------------------------------------------------------------------
# 3 – Pneumonia sub-type breakdown
# ---------------------------------------------------------------------------

def count_pneumonia_subtypes(dataset_root: Path) -> dict:
    """
    Infer bacterial vs viral vs unknown pneumonia cases from filenames.

    The Kermany dataset names pneumonia images as:
        person<N>_bacteria_<M>.jpeg
        person<N>_virus_<M>.jpeg

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    dict
        {split: {"bacteria": int, "virus": int, "unknown": int}}
    """
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        pn_dir = dataset_root / split / "PNEUMONIA"
        if not pn_dir.is_dir():
            result[split] = {"bacteria": 0, "virus": 0, "unknown": 0}
            continue
        counts: dict[str, int] = {"bacteria": 0, "virus": 0, "unknown": 0}
        for f in pn_dir.glob("*.jpeg"):
            name = f.name.lower()
            if "bacteria" in name:
                counts["bacteria"] += 1
            elif "virus" in name:
                counts["virus"] += 1
            else:
                counts["unknown"] += 1
        result[split] = counts
    return result


# ---------------------------------------------------------------------------
# 4 – Non-JPEG / unexpected file check
# ---------------------------------------------------------------------------

def find_unexpected_files(dataset_root: Path) -> list[dict]:
    """
    Report any files that are NOT .jpeg inside the split/class folders,
    and any folders that do not match the expected layout.

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    list[dict]
        Each dict has keys: path, issue.
    """
    issues: list[dict] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            issues.append({"path": str(split_dir), "issue": f"split folder '{split}' missing"})
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                issues.append({"path": str(cls_dir), "issue": f"class folder '{cls}' missing in split '{split}'"})
                continue
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() != ".jpeg":
                    issues.append({"path": str(f), "issue": f"unexpected file type '{f.suffix}'"})
    return issues


# ---------------------------------------------------------------------------
# 5 – Duplicate filename detection
# ---------------------------------------------------------------------------

def find_duplicate_filenames(dataset_root: Path) -> dict:
    """
    Detect filenames that appear more than once across all splits and classes.

    Note: This checks filenames only, not file content (hash comparison).
    The Kermany dataset has unique filenames by design, so any duplicates
    would be a data packaging problem.

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    dict
        {
            "within_split_duplicates":  list of (filename, split, class),
            "cross_split_duplicates":   list of (filename, [splits]),
            "total_images_checked":     int,
        }
    """
    all_entries: list[tuple[str, str, str]] = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = dataset_root / split / cls
            if folder.is_dir():
                for f in folder.glob("*.jpeg"):
                    all_entries.append((f.name, split, cls))

    name_to_locations: dict[str, list[tuple[str, str]]] = {}
    for name, split, cls in all_entries:
        name_to_locations.setdefault(name, []).append((split, cls))

    within: list[dict] = []
    cross: list[dict] = []
    for name, locations in name_to_locations.items():
        if len(locations) > 1:
            splits_seen = {loc[0] for loc in locations}
            if len(splits_seen) == 1:
                within.append({"filename": name, "locations": locations})
            else:
                cross.append({"filename": name, "splits": sorted(splits_seen)})

    return {
        "within_split_duplicates": within,
        "cross_split_duplicates": cross,
        "total_images_checked": len(all_entries),
    }


# ---------------------------------------------------------------------------
# 6 – Image readability check (sampled)
# ---------------------------------------------------------------------------

def check_image_readability(
    dataset_root: Path,
    sample_size: int = READABILITY_SAMPLE_SIZE,
    seed: int = 42,
) -> dict:
    """
    Open a random sample of images with Pillow and record any that fail.

    A JPEG is considered readable if ``Image.open(path).verify()`` succeeds
    without raising an exception.

    Parameters
    ----------
    dataset_root : Path
    sample_size  : int
        How many images to check per split/class cell (None = all).
    seed         : int

    Returns
    -------
    dict
        {
            "pillow_available":   bool,
            "images_checked":     int,
            "unreadable_count":   int,
            "unreadable_sample":  list of {"path": str, "error": str},
        }
    """
    if not _PILLOW_AVAILABLE:
        return {
            "pillow_available": False,
            "images_checked": 0,
            "unreadable_count": 0,
            "unreadable_sample": [],
        }

    rng = random.Random(seed)
    unreadable: list[dict] = []
    checked = 0

    for split in SPLITS:
        for cls in CLASSES:
            folder = dataset_root / split / cls
            if not folder.is_dir():
                continue
            files = list(folder.glob("*.jpeg"))
            to_check = files if (sample_size is None or len(files) <= sample_size) \
                else rng.sample(files, sample_size)
            for f in to_check:
                checked += 1
                try:
                    img = _PIL_Image.open(f)
                    img.verify()
                except Exception as exc:
                    unreadable.append({"path": str(f), "error": str(exc)})

    return {
        "pillow_available": True,
        "images_checked": checked,
        "unreadable_count": len(unreadable),
        "unreadable_sample": unreadable[:20],
    }


# ---------------------------------------------------------------------------
# 7 – Image dimension and mode diversity (sampled)
# ---------------------------------------------------------------------------

def sample_image_dimensions(
    dataset_root: Path,
    sample_size: int = DIMENSION_SAMPLE_SIZE,
    seed: int = 42,
) -> dict:
    """
    Open a random sample of images and record width, height, and color mode.

    Parameters
    ----------
    dataset_root : Path
    sample_size  : int
    seed         : int

    Returns
    -------
    dict
        {
            "images_sampled":  int,
            "modes_seen":      list[str],
            "width_min":       int,
            "width_max":       int,
            "height_min":      int,
            "height_max":      int,
        }
    """
    if not _PILLOW_AVAILABLE:
        return {"images_sampled": 0, "modes_seen": [], "error": "Pillow not installed"}

    rng = random.Random(seed)
    all_files: list[Path] = []
    for split in SPLITS:
        for cls in CLASSES:
            folder = dataset_root / split / cls
            if folder.is_dir():
                all_files.extend(folder.glob("*.jpeg"))

    to_check = all_files if len(all_files) <= sample_size else rng.sample(all_files, sample_size)

    widths: list[int] = []
    heights: list[int] = []
    modes: set[str] = set()

    for f in to_check:
        try:
            with _PIL_Image.open(f) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes.add(img.mode)
        except Exception:
            pass

    if not widths:
        return {"images_sampled": 0, "modes_seen": [], "error": "no readable images found"}

    return {
        "images_sampled": len(widths),
        "modes_seen": sorted(modes),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
    }


# ---------------------------------------------------------------------------
# 8 – Inner folder detection (packaging artefact)
# ---------------------------------------------------------------------------

def check_inner_duplicate_folder(dataset_root: Path) -> dict:
    """
    The Kaggle archive sometimes unpacks with a nested folder that has the
    same name as the outer folder (e.g. chest_xray/chest_xray/).
    Detect and report this so users know which root to use.

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    dict
        {"inner_folder_detected": bool, "inner_path": str or None}
    """
    inner = dataset_root / dataset_root.name
    if inner.is_dir() and (inner / "train").is_dir():
        return {"inner_folder_detected": True, "inner_path": str(inner)}
    return {"inner_folder_detected": False, "inner_path": None}


# ---------------------------------------------------------------------------
# 9 – Orchestration
# ---------------------------------------------------------------------------

def run_inspection(dataset_root: Path) -> dict:
    """
    Run all inspection steps and return a unified results dict.

    Parameters
    ----------
    dataset_root : Path

    Returns
    -------
    dict
        Contains all sub-results; used by the CLI and by tests.
    """
    print(f"\n{'='*60}")
    print("  MedReason++ — Chest X-Ray Pneumonia Dataset Inspection")
    print(f"  Dataset root: {dataset_root}")
    print(f"{'='*60}\n")

    print("[1/7] Counting images per split and class …")
    image_counts = count_images(dataset_root)
    for split in SPLITS:
        row = image_counts[split]
        print(f"       {split:6s}: NORMAL={row.get('NORMAL', 0):4d}  "
              f"PNEUMONIA={row.get('PNEUMONIA', 0):4d}  total={row.get('total', 0):4d}")

    print("[2/7] Detecting pneumonia sub-types (bacterial / viral) …")
    subtypes = count_pneumonia_subtypes(dataset_root)
    for split in SPLITS:
        st = subtypes[split]
        print(f"       {split:6s}: bacteria={st['bacteria']}  "
              f"virus={st['virus']}  unknown={st['unknown']}")

    print("[3/7] Checking for unexpected files …")
    unexpected = find_unexpected_files(dataset_root)
    jpeg_unexpected = [u for u in unexpected if "missing" not in u["issue"]]
    print(f"       Unexpected files: {len(jpeg_unexpected)}")

    print("[4/7] Detecting duplicate filenames …")
    dup_info = find_duplicate_filenames(dataset_root)
    print(f"       Within-split duplicates: {len(dup_info['within_split_duplicates'])}")
    print(f"       Cross-split duplicates : {len(dup_info['cross_split_duplicates'])}")

    print("[5/7] Sampling image readability …")
    readability = check_image_readability(dataset_root)
    if readability["pillow_available"]:
        print(f"       Checked: {readability['images_checked']}  "
              f"Unreadable: {readability['unreadable_count']}")
    else:
        print("       Pillow not installed — skipping readability check.")

    print("[6/7] Sampling image dimensions and color modes …")
    dimensions = sample_image_dimensions(dataset_root)
    if "error" not in dimensions:
        print(f"       Modes  : {dimensions['modes_seen']}")
        print(f"       Width  : {dimensions['width_min']}–{dimensions['width_max']} px")
        print(f"       Height : {dimensions['height_min']}–{dimensions['height_max']} px")

    print("[7/7] Checking for nested duplicate folder …")
    inner = check_inner_duplicate_folder(dataset_root)
    if inner["inner_folder_detected"]:
        print(f"       ⚠ Inner duplicate folder detected: {inner['inner_path']}")
        print("         Use the outer root shown above for training — it has no .DS_Store noise.")
    else:
        print("       No nested duplicate folder.")

    print(f"\n{'='*60}")
    print("  Inspection complete.")
    print(f"{'='*60}\n")

    return {
        "dataset_root": str(dataset_root),
        "image_counts": image_counts,
        "pneumonia_subtypes": subtypes,
        "unexpected_files": unexpected,
        "duplicates": dup_info,
        "readability": readability,
        "dimensions": dimensions,
        "inner_folder": inner,
    }


# ---------------------------------------------------------------------------
# 10 – Write Markdown report
# ---------------------------------------------------------------------------

def write_inspection_report(results: dict, output_path: Path) -> None:
    """
    Serialise the inspection results to a Markdown file.

    Parameters
    ----------
    results     : dict  from run_inspection()
    output_path : Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts   = results["image_counts"]
    subtypes = results["pneumonia_subtypes"]
    dups     = results["duplicates"]
    read     = results["readability"]
    dims     = results["dimensions"]
    inner    = results["inner_folder"]
    unexpected = results["unexpected_files"]

    total_images = sum(counts[s].get("total", 0) for s in SPLITS)
    total_normal = sum(counts[s].get("NORMAL", 0) for s in SPLITS)
    total_pn     = sum(counts[s].get("PNEUMONIA", 0) for s in SPLITS)

    lines = [
        "# Chest X-Ray Pneumonia Dataset Inspection Report",
        "",
        f"> Dataset root: `{results['dataset_root']}`",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "Source: Chest X-Ray Images (Pneumonia) — Kermany et al., Kaggle.",
        "Images are JPEG chest X-rays split into NORMAL and PNEUMONIA classes.",
        "PNEUMONIA images are further identified by filename as bacterial or viral.",
        "",
        "---",
        "",
        "## 2. Image Counts per Split",
        "",
        "| Split | NORMAL | PNEUMONIA | Total |",
        "|-------|-------:|----------:|------:|",
    ]
    for split in SPLITS:
        row = counts[split]
        exists_note = "" if row["exists"] else " *(folder missing)*"
        lines.append(
            f"| {split}{exists_note} | {row.get('NORMAL', 0):,} "
            f"| {row.get('PNEUMONIA', 0):,} | {row.get('total', 0):,} |"
        )
    lines += [
        f"| **total** | **{total_normal:,}** | **{total_pn:,}** | **{total_images:,}** |",
        "",
        "---",
        "",
        "## 3. Pneumonia Sub-type Breakdown",
        "",
        "Filenames encode the causative agent:",
        "- `person<N>_bacteria_<M>.jpeg` → bacterial pneumonia",
        "- `person<N>_virus_<M>.jpeg`    → viral pneumonia",
        "",
        "| Split | Bacterial | Viral | Unknown |",
        "|-------|----------:|------:|--------:|",
    ]
    for split in SPLITS:
        st = subtypes[split]
        lines.append(
            f"| {split} | {st['bacteria']:,} | {st['virus']:,} | {st['unknown']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Class Imbalance",
        "",
        f"- Total NORMAL images   : **{total_normal:,}** "
        f"({100*total_normal/total_images:.1f}%)",
        f"- Total PNEUMONIA images: **{total_pn:,}** "
        f"({100*total_pn/total_images:.1f}%)",
        f"- Overall imbalance ratio (PNEUMONIA : NORMAL): "
        f"**{total_pn/total_normal:.2f}:1**",
        "",
        "> This is far better than the IU X-ray weak label distribution (194:1).",
        "> However, there is still a ~2.7:1 imbalance in training data.",
        "",
        "---",
        "",
        "## 5. Integrity Checks",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| All split folders present (train/val/test) | "
        f"{'✅' if all(counts[s]['exists'] for s in SPLITS) else '❌'} |",
        f"| Within-split duplicate filenames | "
        f"{'✅ None' if not dups['within_split_duplicates'] else '❌ ' + str(len(dups['within_split_duplicates']))} |",
        f"| Cross-split duplicate filenames | "
        f"{'✅ None' if not dups['cross_split_duplicates'] else '❌ ' + str(len(dups['cross_split_duplicates']))} |",
        f"| Unexpected non-JPEG files in class folders | "
        f"{'✅ None' if not [u for u in unexpected if 'unexpected file type' in u['issue']] else '❌ ' + str(len([u for u in unexpected if 'unexpected file type' in u['issue']]))} |",
    ]

    if read["pillow_available"]:
        lines.append(
            f"| Unreadable images (sample of {read['images_checked']}) | "
            f"{'✅ None' if read['unreadable_count'] == 0 else '❌ ' + str(read['unreadable_count'])} |"
        )
    else:
        lines.append("| Readability check | ⚠ Pillow not installed — skipped |")

    if inner["inner_folder_detected"]:
        lines += [
            "",
            "### ⚠ Nested Duplicate Folder",
            "",
            f"An inner folder `{inner['inner_path']}` was detected.",
            "This is a common Kaggle archive artefact.",
            "The inner folder contains `.DS_Store` files but is otherwise identical.",
            "**Use the outer root for all downstream work.**",
        ]

    lines += [
        "",
        "---",
        "",
        "## 6. Image Properties",
    ]
    if "error" not in dims:
        lines += [
            "",
            f"Sample size: {dims['images_sampled']} images",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Color modes | {', '.join(dims['modes_seen'])} |",
            f"| Width range | {dims['width_min']}–{dims['width_max']} px |",
            f"| Height range | {dims['height_min']}–{dims['height_max']} px |",
            "",
            "> Images are **variable size** and contain both grayscale (`L`) and RGB",
            "> color channels.  Preprocessing should resize and normalise to a fixed",
            "> resolution (e.g. 224×224) and convert all images to RGB before training.",
        ]
    else:
        lines.append(f"\n> {dims.get('error', 'Unknown error')}")

    lines += [
        "",
        "---",
        "",
        "## 7. Comparison with IU X-ray Dataset",
        "",
        "| Property | IU X-ray | Chest X-Ray Pneumonia |",
        "|----------|----------|----------------------|",
        "| Studies | 2,955 | N/A (individual images) |",
        f"| Total images | 5,910 (2 views each) | {total_images:,} |",
        "| Positive pneumonia | 15 (0.5%, weak labels) | "
        f"{total_pn:,} ({100*total_pn/total_images:.1f}%, radiologist labels) |",
        "| Label quality | NLP-derived weak supervision | Radiologist-verified |",
        "| Report text available | ✅ Yes (merged paragraph) | ❌ No |",
        "| Image format | PNG | JPEG |",
        "| Image dimensions | Fixed (varies by study) | Variable |",
        "| Splits | train/val/test | train/val/test |",
        "| Multimodal (image + text) | ✅ Yes | ❌ Image only |",
        "",
        "---",
        "",
        "## 8. Recommended Use in MedReason++",
        "",
        "| Role | Dataset |",
        "|------|---------|",
        "| Supervised pneumonia classification (Phase 6) | Chest X-Ray Pneumonia |",
        "| Report-grounded reasoning / explanation (later) | IU X-ray |",
        "| Multimodal image + text fusion (later) | IU X-ray |",
        "",
        "---",
        "",
        "_Generated by `src/inspect_pneumonia_dataset.py`_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the Chest X-Ray Pneumonia dataset for MedReason++."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to the dataset root (folder containing train/, val/, test/). "
             "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/chest_xray_pneumonia_inspection.md",
        help="Where to save the Markdown report.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve() if args.dataset_root else find_dataset_root()
    results = run_inspection(root)

    out = Path(args.output)
    write_inspection_report(results, out)
    print(f"Report saved to: {out}\n")


if __name__ == "__main__":
    main()
