# MedReason++
## Evidence-Grounded Multimodal Clinical Reasoning for Chest X-ray Pneumonia Detection

---

## Web Application

The MedReason++ research interface combines chest X-ray analysis, optional radiology report text, patient context, visual activation maps, and evidence checks in one workspace. It is a research prototype and is not validated for clinical use.

### Run locally

```powershell
python -m pip install -r requirements.txt
cd webapp
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. To enable the optional Gemini reasoning agents, set `GEMINI_API_KEY` in the environment before starting the server. Without it, image analysis and visual explanations can run independently.

Trained model checkpoints, local virtual environments, and image datasets are intentionally not included in this repository. For full image analysis, place the trained checkpoints at the paths expected by `src/extract_features.py` and configure any required datasets locally. See [the web application guide](webapp/README.md) for interface details.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Dataset Inspection | ✅ Complete |
| Phase 2 | Preprocessing & Feature Extraction | 🔜 Pending |
| Phase 3 | Model Development | 🔜 Pending |
| Phase 4 | Evaluation | 🔜 Pending |

---

## Phase 1 – Dataset Inspection

### Dataset

The project uses the **IU X-ray dataset** in R2Gen format, stored at:

```
iu_xray/iu_xray/
    annotation.json          ← 2,955 radiology reports with image references
    images/
        <CXR_ID>/
            0.png            ← frontal (PA) view
            1.png            ← lateral view
```

### How to Run the Dataset Inspection

**Requirements:** Python 3.9+, no third-party packages needed for inspection alone.

```bash
# From the project root
python src/inspect_dataset.py --data_root iu_xray/iu_xray
```

This will:
1. Load and validate `annotation.json`
2. Count all records per split (train / val / test)
3. Analyse report text (word counts, de-identification markers)
4. Count PNG files on disk and compare to annotation references
5. Check for missing files, orphaned images, and duplicate records
6. Verify the image ↔ report naming convention
7. Write a full summary to `reports/dataset_inspection.md`

**Custom output path:**

```bash
python src/inspect_dataset.py \
    --data_root iu_xray/iu_xray \
    --output reports/dataset_inspection.md
```

**Auto-detection (no flags needed if dataset is in the default location):**

```bash
python src/inspect_dataset.py
```

### How to Run the Tests

```bash
# Unit tests only (no dataset needed — uses synthetic fixture data)
pytest tests/test_dataset_inspection.py -v

# Integration tests run automatically when the real dataset is present
pytest tests/test_dataset_inspection.py -v
```

All 34 tests should pass. The 6 integration tests (prefixed `TestRealDatasetIntegration`)
run only if the real dataset is found; they are automatically skipped otherwise.

### Inspection Results (Quick Summary)

| Item | Value |
|------|-------|
| Total records | 2,955 |
| Training | 2,069 |
| Validation | 296 |
| Test | 590 |
| Total PNG files | 6,091 |
| Missing image files | 0 |
| Duplicate records | 0 |
| Reports with XXXX tokens | 1,227 |
| Empty reports | 0 |

See [`reports/dataset_inspection.md`](reports/dataset_inspection.md) for the full report.

---

## Project Structure

```
MedReasonPlus/
├── iu_xray/
│   └── iu_xray/
│       ├── annotation.json
│       └── images/
│           └── <CXR_ID>/
│               ├── 0.png
│               └── 1.png
├── src/
│   └── inspect_dataset.py     ← Phase 1 inspection script
├── tests/
│   └── test_dataset_inspection.py
├── reports/
│   └── dataset_inspection.md  ← auto-generated inspection summary
└── README.md
```
