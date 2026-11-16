# Pneumonia Labeling Summary

> **⚠ DISCLAIMER:** These labels are *weak supervision* derived from
> radiology report text using rule-based NLP patterns. They are
> **NOT verified clinical ground truth** and must NOT be used as
> reliable training labels without expert clinical review.
> A significant fraction of reports use hedged or proxy language.
> Always validate a representative sample before downstream use.

> Input CSV: `processed\iu_xray_index.csv`
> Output CSV: `reports\pneumonia_labeling_summary.md`

---

## 1. Label Counts

| Label | Studies | Percentage |
|-------|--------:|-----------:|
| `positive` | 15 | 0.5% |
| `negative` | 2,914 | 98.6% |
| `uncertain` | 26 | 0.9% |
| **total** | **2,955** | 100% |

---

## 2. Confidence Breakdown

| Label | Confidence | Count |
|-------|-----------|------:|
| `positive` | HIGH | 5 |
| `positive` | MEDIUM | 10 |
| `negative` | HIGH | 176 |
| `negative` | MEDIUM | 1,398 |
| `negative` | LOW | 1,340 |
| `uncertain` | MEDIUM | 11 |
| `uncertain` | LOW | 15 |

---

## 3. Label Distribution by Split

| Split | positive | negative | uncertain |
|-------|--------:|--------:|--------:|
| train | 13 | 2,037 | 19 |
| val | 0 | 293 | 3 |
| test | 2 | 584 | 4 |

---

## 4. Most Common Labeling Rules

| Rule | Count |
|------|------:|
| `no_focal_consolidation` | 1,398 |
| `no_relevant_language_found` | 1,340 |
| `no_opacity_to_suggest_pneumonia` | 123 |
| `no_pneumonia_explicit` | 52 |
| `uncertain_hedged_consolidation` | 15 |
| `proxy_airspace_disease_positive` | 9 |
| `uncertain_atelectasis_vs_pneumonia` | 8 |
| `pneumonia_direct_assertion` | 4 |
| `uncertain_could_represent_pneumonia` | 3 |
| `pneumonia_seen_confirmed` | 1 |
| `no_pneumonic_consolidation_clear` | 1 |
| `pneumonia_suggestive_probable` | 1 |

---

## 5. Representative Examples

### Positive examples

| Report ID | Rule | Confidence | Report excerpt |
|-----------|------|-----------|----------------|
| `CXR2859_IM-1266` | `pneumonia_direct_assertion` | HIGH | There is a rounded dense opacity in the lateral left midlung zone probably the left upper lobe most suggestive of a rounded pneumonia. There is no ple… |
| `CXR1072_IM-0052-1001` | `proxy_airspace_disease_positive` | MEDIUM | The heart and mediastinal contours are stable. Aorta is calcified and tortuous, compatible with atherosclerotic disease. Since the prior study, there'… |
| `CXR25_IM-1024` | `proxy_airspace_disease_positive` | MEDIUM | The heart is within normal limits in size. Surgical suture material projects over the right lung apex. The lungs are hyperlucent and hyperinflated com… |

### Negative examples

| Report ID | Rule | Confidence | Report excerpt |
|-----------|------|-----------|----------------|
| `CXR2384_IM-0942` | `no_opacity_to_suggest_pneumonia` | HIGH | The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No… |
| `CXR2926_IM-1328` | `no_focal_consolidation` | MEDIUM | Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact.… |
| `CXR1451_IM-0291` | `no_relevant_language_found` | LOW | Left lower lobe calcified granuloma. Heart size normal. No pleural effusion or pneumothorax. Mild medial right atelectasis. Mild emphysema.… |

### Uncertain examples

| Report ID | Rule | Confidence | Report excerpt |
|-----------|------|-----------|----------------|
| `CXR2338_IM-0905` | `uncertain_hedged_consolidation` | LOW | Compared to prior examination, there is significant improvement in aeration bilaterally, with improved bilateral airspace opacities. Currently, there … |
| `CXR343_IM-1658` | `uncertain_hedged_consolidation` | LOW | Heart size and mediastinal contour are normal. Pulmonary vascularity is normal. is not diffuse interstitial prominence, which has chronic appearance. … |
| `CXR3813_IM-1922` | `uncertain_hedged_consolidation` | LOW | There is XXXX opacity left lung base may represent atelectasis or early infiltrate. The cardiomediastinal silhouette is within normal limits. No pneum… |

---

## 6. Labeling Methodology

The pipeline applies a ranked list of regex rules to each report:

| Tier | Direction | Examples |
|------|-----------|----------|
| 1 | Positive / HIGH | "there is a right lower lobe pneumonia", "airspace disease consistent with pneumonia" |
| 2 | Positive / MEDIUM | "concerning for underlying pneumonia", "most suggestive of a rounded pneumonia" |
| 3 | Negative / HIGH | "no pneumonia", "no focal air space opacity to suggest a pneumonia" |
| 4 | Negative / MEDIUM | "no focal consolidation", "lungs are clear" |
| 5 | Uncertain / MEDIUM | "could represent a pneumonia", "atelectasis versus pneumonia" |
| 6 | Uncertain / LOW | "cannot exclude", "question infiltrate" |

**Conflict resolution:** If both positive and negative rules match,
the higher-confidence direction wins unless they are within one tier of
each other, in which case the report is marked `uncertain`.

**Fallback:** If no pneumonia-relevant language is found at all, the
report is marked `negative / LOW` (clean chest by default).

---

## 7. Phase 3 Checkpoint

| Criterion | Status |
|-----------|--------|
| `processed/iu_xray_labeled.csv` created | ✅ |
| Three labels used (positive/negative/uncertain) | ✅ |
| Explicit negation handled | ✅ |
| 'no X to suggest pneumonia' treated as negative | ✅ |
| Hedged language → uncertain | ✅ |
| Original CSV unmodified | ✅ |
| Disclaimer included | ✅ |
| No invented labels from filenames | ✅ |

---

_Generated by `src/label_pneumonia_reports.py`_