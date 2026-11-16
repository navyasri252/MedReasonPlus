# Phase 4 — Label Audit Summary

> **⚠ DISCLAIMER:** Labels are *weak supervision* derived from NLP patterns.
> They have **NOT been clinically validated**.
> This audit is a project-level review only.

> Audit CSV: `processed\label_audit.csv`

---

## 1. Label Counts

| Label | Count | % of total |
|-------|------:|-----------:|
| positive  | 15 | 0.51% |
| negative  | 2,914 | 98.61% |
| uncertain | 26 | 0.88% |
| **total** | **2,955** | 100% |

---

## 2. Label Distribution by Split

| Split | positive | negative | uncertain | total |
|-------|--------:|--------:|--------:|------:|
| train | 13 | 2,037 | 19 | 2,069 |
| val | 0 | 293 | 3 | 296 |
| test | 2 | 584 | 4 | 590 |

> Val split has **0 positive** studies. This means a model trained with
> standard train/val evaluation would never see a positive validation example
> during training — validation loss would be uninformative for the positive class.

---

## 3. Class Imbalance Analysis

**Negative-to-positive ratio: 194.3:1**

The positive class makes up only **0.51%** of all studies.

### Why accuracy alone is misleading

> A model that predicts 'negative' for every study would be 98.6% accurate but would recall 0% of the 15 positive cases.  Metrics like precision, recall, F1, and AUC-ROC are required to evaluate a model on imbalanced data.

| Metric | Naive always-negative classifier |
|--------|----------------------------------|
| Accuracy | 98.61% |
| Recall (positive class) | 0% |
| Precision (positive class) | undefined (no positives predicted) |
| F1 (positive class) | 0 |

### Recommended metrics for this problem

- **Recall (sensitivity)** — did the model find the positive cases?
- **Precision** — when it predicted positive, was it right?
- **F1-score (positive class)** — harmonic mean of precision and recall.
- **AUC-ROC / AUC-PR** — threshold-independent performance.
- **Confusion matrix** — inspect false-negative rate separately.

### Strategies to address imbalance (for Phase 5+)

1. **Weighted loss** — assign higher loss weight to the positive class.
2. **Oversampling** — use `SMOTE` or repeat positive samples during training.
3. **Undersampling** — randomly drop negative training samples.
4. **Threshold tuning** — move the decision threshold to favour recall.
5. **Use uncertain as a third class** — do not force uncertain into binary.
6. **Seek a more balanced dataset** — the IU X-ray dataset was not designed
   for pneumonia detection; a dataset curated for pneumonia would be better.

---

## 4. Integrity Checks

| Check | Result |
|-------|--------|
| Cross-split contamination | ✅ None |
| View-label consistency | ✅ All consistent |
| IDs in labeled not in index | ✅ None |
| IDs in index not in labeled | ✅ None |

---

## 5. Audit CSV Contents

Audit CSV: `processed\label_audit.csv`

| Reason | Count |
|--------|------:|
| All positive studies | 15 |
| All uncertain studies | 26 |
| Random negative sample (seed=42) | 100 |
| **Total rows in audit CSV** | **141** |

---

## 6. Positive Label Breakdown

| Labeling Rule | Count |
|---------------|------:|
| `proxy_airspace_disease_positive` | 9 |
| `pneumonia_direct_assertion` | 4 |
| `pneumonia_suggestive_probable` | 1 |
| `pneumonia_seen_confirmed` | 1 |

### All positive studies

| Report ID | Split | Rule | Confidence | Report excerpt (first 200 chars) |
|-----------|-------|------|-----------|----------------------------------|
| `CXR1193_IM-0129` | test | `proxy_airspace_disease_positive` | MEDIUM | Stable enlarged cardiac silhouette. Persistent bilateral lower lobe airspace disease, not significantly XXXX compared to prior. No pleural effusion or pneumothorax. No acute bony abnormality.… |
| `CXR3411_IM-1649-0001` | test | `proxy_airspace_disease_positive` | MEDIUM | There are low lung volumes with bronchovascular crowding. There is patchy left lower lobe airspace disease. There are XXXX opacities in the right mid lung, XXXX subsegmental atelectasis. No significan… |
| `CXR1072_IM-0052-1001` | train | `proxy_airspace_disease_positive` | MEDIUM | The heart and mediastinal contours are stable. Aorta is calcified and tortuous, compatible with atherosclerotic disease. Since the prior study, there's been interval development of left lower lobe air… |
| `CXR1691_IM-0453` | train | `proxy_airspace_disease_positive` | MEDIUM | There are persistent low lung volumes. There is stable streaky left lower lobe airspace disease. Probable XXXX residual left pneumothorax. No large pleural effusion. Stable cardiomediastinal contour. … |
| `CXR2062_IM-0699-0001` | train | `proxy_airspace_disease_positive` | MEDIUM | Right subclavian catheter, distal tip posterior to the head of the clavicle, the level of the subclavian vein. Low lung volumes. No pleural effusion. Left lower lobe airspace disease, XXXX atelectasis… |
| `CXR245_IM-0985` | train | `proxy_airspace_disease_positive` | MEDIUM | XXXX XXXX and lateral chest examination was obtained. The heart silhouette is normal in size and contour. Aortic XXXX appear unremarkable. Lungs demonstrate left upper lobe airspace disease most XXXX … |
| `CXR25_IM-1024` | train | `proxy_airspace_disease_positive` | MEDIUM | The heart is within normal limits in size. Surgical suture material projects over the right lung apex. The lungs are hyperlucent and hyperinflated compatible with emphysema. There is left lower lobe a… |
| `CXR2761_IM-1208` | train | `pneumonia_suggestive_probable` | MEDIUM | Stable, normal cardiac size, mediastinum, and central pulmonary vasculature. Interval development of mild patchy airspace opacities within the posterior aspect of the right upper lobe, concerning for … |
| `CXR2859_IM-1266` | train | `pneumonia_direct_assertion` | HIGH | There is a rounded dense opacity in the lateral left midlung zone probably the left upper lobe most suggestive of a rounded pneumonia. There is no pleural effusion. The heart and mediastinum are norma… |
| `CXR3069_IM-1432` | train | `pneumonia_direct_assertion` | HIGH | Right upper lobe airspace disease consistent with pneumonia given patient's history. The lungs are otherwise clear. No pleural effusions or pneumothoraces. Heart and mediastinum of normal size and con… |
| `CXR3078_IM-1438` | train | `pneumonia_direct_assertion` | HIGH | The cardiomediastinal silhouette is normal size and configuration. Pulmonary vasculature within normal limits. There is a right lower lobe pneumonia. No pleural effusion. No pneumothorax.… |
| `CXR3103_IM-1454` | train | `pneumonia_seen_confirmed` | HIGH | Cardiomediastinal silhouette stable and unremarkable. Stable eventration of the right hemidiaphragm. There is redemonstration without significant interval change of mild subsegmental atelectasis of th… |
| `CXR564_IM-2165` | train | `proxy_airspace_disease_positive` | MEDIUM | The cardiomediastinal silhouette and vasculature are within normal limits for size and contour. There is right upper lobe airspace disease.. There is a rounded nodular opacity in the left upper lung m… |
| `CXR808_IM-2341` | train | `pneumonia_direct_assertion` | HIGH | Overall low lung lines. There is scarring or subsegmental atelectasis at the right lung base. In the left lower lobe there is airspace disease consistent with pneumonia. No pneumothorax. Heart and med… |
| `CXR959_IM-2449` | train | `proxy_airspace_disease_positive` | MEDIUM | The heart and mediastinal contours are stable. There is minimal patchy right lower lobe airspace disease identified. No pleural effusion or pneumothorax.… |

---

## 7. Uncertain Label Breakdown

| Labeling Rule | Count |
|---------------|------:|
| `uncertain_hedged_consolidation` | 15 |
| `uncertain_atelectasis_vs_pneumonia` | 8 |
| `uncertain_could_represent_pneumonia` | 3 |

### All uncertain studies

| Report ID | Split | Rule | Confidence | Report excerpt (first 200 chars) |
|-----------|-------|------|-----------|----------------------------------|
| `CXR3445_IM-1668` | test | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | No gross consolidation, atelectasis or infiltrate. No pleural fluid collection or pneumothorax. Cardiomediastinal silhouette is within normal limits. XXXX XXXX is intact.… |
| `CXR3721_IM-1859` | test | `uncertain_hedged_consolidation` | LOW | Streaky opacity is noted within the left lung base which may represent focal area of atelectasis. Right lung is grossly clear. Cardiac silhouette and mediastinal contours are within normal limits. The… |
| `CXR569_IM-2169-0001` | test | `uncertain_hedged_consolidation` | LOW | The heart size and pulmonary vascularity appear within normal limits. The left hemidiaphragm remains elevated. Right base densities are again noted which appear improved. Previously seen left pleural … |
| `CXR932_IM-2430` | test | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | No gross consolidation, atelectasis or infiltrate. No pleural fluid collection or pneumothorax. Cardiomediastinal silhouette is within normal limits. XXXX XXXX is intact.… |
| `CXR1075_IM-0054` | train | `uncertain_hedged_consolidation` | LOW | Lungs are hyperinflated with interstitial changes of severe emphysema. There is an ill-defined pleural parenchymal opacity in the left upper lobe. This may represent scarring but is incompletely evalu… |
| `CXR1257_IM-0174` | train | `uncertain_hedged_consolidation` | LOW | Mild cardiomegaly. Hypoinflation of the lungs. Right basilar XXXX opacity may represent atelectasis. Lungs are otherwise clear without focal consolidation, pleural effusion, or pneumothorax. Sclerosis… |
| `CXR1462_IM-0299` | train | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | No gross consolidation, atelectasis or infiltrate. No pleural fluid collection or pneumothorax. Cardiomediastinal silhouette is within normal limits. XXXX XXXX is intact.… |
| `CXR1524_IM-0339` | train | `uncertain_hedged_consolidation` | LOW | Heart size is normal. The aorta is tortuous, and cannot exclude ascending aortic aneurysm. The pulmonary vascularity is normal. There residual to prior granulomatous infection. Lungs are otherwise cle… |
| `CXR1704_IM-0464` | train | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | There is blunting of the left costophrenic XXXX compatible with a moderate to large left pleural fluid collection. There are areas of airspace opacity within the left lung base which may represent ate… |
| `CXR1740_IM-0488` | train | `uncertain_hedged_consolidation` | LOW | Frontal and lateral views of the chest show an unchanged cardiomediastinal silhouette. There is XXXX airspace opacity in the mid right lung radiating from the right hilum to the pleura and bordered in… |
| `CXR183_IM-0537` | train | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | Heart size within normal limits. There are low lung volumes with bronchovascular crowding. There is mild increased airspace opacity within the right lung base which may represent atelectasis or infilt… |
| `CXR2036_IM-0680` | train | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | There is mild cardiomegaly. The thoracic aorta is tortuous. Lung volumes are low with asymmetric elevation of the right hemidiaphragm. There is platelike atelectasis in the right midlung along with bi… |
| `CXR2165_IM-0781-0001` | train | `uncertain_hedged_consolidation` | LOW | Low lung volumes are present. The heart size and pulmonary vascularity appear within normal limits. There has been interval development of bibasilar opacities. The appearance of the right base opacity… |
| `CXR2338_IM-0905` | train | `uncertain_hedged_consolidation` | LOW | Compared to prior examination, there is significant improvement in aeration bilaterally, with improved bilateral airspace opacities. Currently, there are only minimal streaky opacities in the bilatera… |
| `CXR2415_IM-0961` | train | `uncertain_hedged_consolidation` | LOW | The heart is mildly enlarged. The lungs are hypoinflated with mildly elevated left hemidiaphragm. There is patchy opacity in the left lung base which may be secondary to atelectasis and/or possible in… |
| `CXR3260_IM-1547` | train | `uncertain_hedged_consolidation` | LOW | 1. Cardiomegaly and/or pericardial effusion. 2. Right base opacity XXXX combination of pleural effusion and atelectasis/airspace disease. Cannot exclude elevation right hemidiaphragm. 3. Left lung rel… |
| `CXR332_IM-1588` | train | `uncertain_could_represent_pneumonia` | MEDIUM | There are are streaky retrocardiac left lower lobe opacities, in the correct clinical setting this could represent a pneumonia. There is no pneumothorax or pleural effusion. The cardiac silhouette is … |
| `CXR343_IM-1658` | train | `uncertain_hedged_consolidation` | LOW | Heart size and mediastinal contour are normal. Pulmonary vascularity is normal. is not diffuse interstitial prominence, which has chronic appearance. Cannot exclude early pulmonary edema. Two airspace… |
| `CXR3813_IM-1922` | train | `uncertain_hedged_consolidation` | LOW | There is XXXX opacity left lung base may represent atelectasis or early infiltrate. The cardiomediastinal silhouette is within normal limits. No pneumothorax or pleural effusion.… |
| `CXR3882_IM-1970` | train | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | Eventration of the left diaphragm is noted. Question left basilar atelectasis versus infiltrate. No evidence of pneumothorax. Generalized lung volumes. No definite pleural effusions. Heart size within… |
| `CXR411_IM-2056` | train | `uncertain_hedged_consolidation` | LOW | There is some minimal patchy opacity in left base which may represent atelectasis or scarring. The lungs are otherwise clear. The heart and mediastinum are normal for age. There is some arthritic chan… |
| `CXR590_IM-2185` | train | `uncertain_could_represent_pneumonia` | MEDIUM | The cardiomediastinal silhouette is normal size and configuration. Pulmonary vasculature within normal limits. There is right middle lobe airspace disease, may reflect atelectasis or pneumonia. No ple… |
| `CXR594_IM-2187` | train | `uncertain_hedged_consolidation` | LOW | There are T-spine osteophytes. There calcified costochondral cartilages. There is loss of disc XXXX of a midthoracic vertebral body. There are streaky opacities in both lung bases which may represent … |
| `CXR1720_IM-0475-0001` | val | `uncertain_hedged_consolidation` | LOW | There are very low lung volumes with associated central bronchovascular crowding. There is elevation of the left hemidiaphragm. There are XXXX-filled loops of mildly dilated colon in the left upper qu… |
| `CXR2961_IM-1355` | val | `uncertain_atelectasis_vs_pneumonia` | MEDIUM | There is patchy opacity in the left base concerning for atelectasis versus pneumonitis. There is a curvilinear lucency that appears to be in the right hemidiaphragm and a small amount of free intraper… |
| `CXR3074_IM-1435` | val | `uncertain_could_represent_pneumonia` | MEDIUM | Sequelae of old granulomatous disease. Vague patchy opacity in the right midlung which may represent early pneumonia. No pleural effusions or pneumothoraces. Heart and mediastinum of normal size and c… |

---

## 8. How to Use the Audit CSV

The file `processed/label_audit.csv` is ready for human review.
Open it in Excel, Google Sheets, or any CSV editor.

For each row, a reviewer should:

1. Read the `report` column carefully.
2. Fill in `manual_review_label` with one of: `positive`, `negative`, `uncertain`, `exclude`.
   - `exclude` means the report is too ambiguous, corrupted, or not a chest X-ray report.
3. Add any clinical notes or concerns in `reviewer_notes`.
4. Pay special attention to rows with `labeling_confidence = MEDIUM` or `LOW`.

**Priority order for review:**
1. All `positive` rows — these are the signal class; false positives matter most.
2. All `uncertain` rows — these may be re-classifiable or should be excluded.
3. The 100 `negative_sample` rows — spot-check for false negatives.

---

## 9. Phase 4 Checkpoint

| Criterion | Status |
|-----------|--------|
| `processed/label_audit.csv` created | ✅ |
| All positive studies included | ✅ (15) |
| All uncertain studies included | ✅ (26) |
| 100 negative sample included (seed=42) | ✅ |
| manual_review_label column present (blank) | ✅ |
| reviewer_notes column present (blank) | ✅ |
| Label counts by split reported | ✅ |
| Cross-split contamination check | ✅ |
| View-label consistency check | ✅ |
| Class imbalance explained | ✅ |
| Why accuracy is misleading explained | ✅ |
| Original CSVs unmodified | ✅ |
| No model training | ✅ |

---

## 10. Decision Gate for Phase 5

Before proceeding to model training, we recommend deciding one of:

| Option | When to choose |
|--------|----------------|
| **Proceed with current labels** | Reviewer confirms ≥10 true positives after manual review |
| **Revise labeling rules** | Manual review reveals systemic false positives/negatives |
| **Use a different dataset** | IU X-ray lacks sufficient positive examples for reliable training |
| **Treat as anomaly detection** | Relabel the 15 positives as anomalies; train on negatives only |

The **recommended choice** given only 15 positives out of 2,955 is either
to **revise the labeling rules** (the proxy rules may be too conservative)
or to **use a pneumonia-specific dataset** such as the CheXpert or
NIH ChestX-ray14 datasets, which have verified radiologist labels.

---

_Generated by `src/audit_pneumonia_labels.py`_