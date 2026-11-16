# Teammate Handoff Guide — MedReason++

**For:** The team member writing the conference paper
**Prepared by:** Project inspection of all code, metrics, and output files

Read this first. The full details are in `research_information_package.md`.

---

## 1. Project Summary (30 seconds)

MedReason++ is a paediatric chest X-ray pneumonia classification system. It trains image classifiers (ResNet-18, ConvNeXt-Tiny, CXformer-S), fuses the two best image models at the feature level (the "Hybrid"), generates GradCAM++ heatmaps for visual explanation, checks predictions against a rule-based text verifier when a radiology report is available, and (optionally) runs three Gemini agents to produce a structured clinical summary.

The best model is the **Hybrid 2-modal (ConvNeXt-Tiny + CXformer-S)** with ROC-AUC 0.988 and sensitivity 99.5% on a 624-image paediatric test set.

---

## 2. Main Research Contribution

The paper's core claim is that **feature-level fusion of a CNN and a chest-domain ViT** (ConvNeXt-Tiny + CXformer-S) achieves higher ROC-AUC (0.988) than either model individually (0.971 and 0.970), while maintaining near-perfect recall (only 2 of 390 pneumonia cases missed).

Secondary contributions include GradCAM++ explainability, a rule-based report-text verification component, and a multi-agent Gemini reasoning architecture. These are implemented but not quantitatively evaluated.

---

## 3. Actual Experimental Findings

**Fully verified (confirmed from local JSON files + CM cross-checks):**

| Model | Accuracy | Recall | Specificity | F1 | AUC |
|-------|---------|--------|------------|-----|-----|
| ResNet-18 (3ep, CPU) | 83.8% | 99.7% | 57.3% | 88.5% | 96.1% |
| ResNet-18 extended (4ep, CPU) | 88.8% | 99.2% | 71.4% | 91.7% | 96.8% |
| **Hybrid 2-modal (Kaggle T4)** | **90.7%** | **99.5%** | 76.1% | **93.1%** | **98.8%** |

**Documented but not locally verified (need to confirm from Kaggle):**

| Model | Accuracy | Recall | Specificity | F1 | AUC |
|-------|---------|--------|------------|-----|-----|
| ConvNeXt-Tiny (Kaggle T4, 8ep) | 85.7% | 99.5% | 62.8% | 89.7% | 97.1% |
| CXformer-S (Kaggle T4, 10ep) | **91.0%** | 95.9% | **82.9%** | **93.0%** | 97.0% |

All results on 624-image test set (234 NORMAL, 390 PNEUMONIA), threshold=0.50.

---

## 4. Key Methodology

### What was trained and evaluated
1. ResNet-18 — full fine-tuning on Kermany, CPU, AdamW
2. ConvNeXt-Tiny — full fine-tuning, ImageNet pretrained, Kaggle T4, AdamW, lr=5e-5, batch=64
3. CXformer-S — **linear probing only** (backbone frozen), Kaggle T4, only head trained, lr=1e-4, batch=32, input=518×518
4. Hybrid 2-modal — both encoders frozen, only projection layers (768→256, 384→256) + classifier head trained, lr=3e-4, batch=32, 8 epochs

### Data split
Stratified 85/15 from the Kermany training folder (seed=42):
- Train: 4,434 | Val: 782 | Test: 624 (never touched until final evaluation)

### What was implemented but NOT trained
- `AdaptiveFusionModel` (4-modality: image + text + metadata) — random weights, no evaluation
- `MetadataEncoder` — random weights, no evaluation
- Gemini agents — no quantitative evaluation

---

## 5. Important Distinctions

### CRITICAL: Do NOT confuse these two models

| | 2-modal Hybrid (evaluated) | 4-modal AdaptiveFusionModel (NOT evaluated) |
|--|--|--|
| Inputs | ConvNeXt features (768d) + CXformer features (384d) | Image + text + metadata |
| Trained? | **Yes** — checkpoint at `processed/hybrid_convnext_cxformer/` | **No** — random weights always |
| Checkpoint | `best_hybrid_model.pth` (201 MB) | None |
| Used for prediction in pipeline? | **Yes** — provides the canonical NORMAL/PNEUMONIA label | No — output discarded |
| Results attributed to it | All Hybrid accuracy/AUC/recall/F1 numbers | None — do not cite |

**The paper may only attribute the reported accuracy/AUC/recall numbers to the 2-modal image-only Hybrid.** The 4-modal architecture is a proposed design that has not been trained.

### The "knowledge graph" is not a graph
`src/knowledge_graph.py` contains a Python list of 12 `Finding` dataclass instances with regex patterns. There is no graph data structure, no nodes, no edges. Use the term **"rule-based evidence verification"** in the paper.

### GradCAM++ was only visually inspected
No quantitative evaluation of explanation quality was performed. Do not claim "GradCAM++ proves the model is clinically reliable."

### Attention weights in the demo UI are meaningless
The percentages shown as "modality contributions" (e.g., "ConvNeXt: 28%, CXformer: 21%") come from the untrained AdaptiveFusionModel with random weights. They have no evidential value.

---

## 6. Important Limitations to State in the Paper

1. Paediatric-only dataset — no adult validation
2. Labels are lung opacity annotations, not confirmed diagnoses
3. 4-modal adaptive fusion architecture is implemented but not trained — do not report its outputs as results
4. GradCAM++ not quantitatively evaluated
5. Gemini agent outputs not evaluated for clinical accuracy
6. No comparison against published methods on this exact test set
7. Distribution shift: val accuracy >> test accuracy in all models (indicates different data sources)

---

## 7. Files the Teammate Should Read (in order)

| Priority | File | Why |
|----------|------|-----|
| 1 | `research_paper/research_information_package.md` | Full verified reference — all details |
| 2 | `research_paper/claim_verification_table.md` | What can and cannot be claimed in the paper |
| 3 | `reports/experiment_results.md` | Manually recorded experiment log |
| 4 | `processed/hybrid_convnext_cxformer/test_metrics.json` | Verified hybrid metrics |
| 5 | `processed/phase6b/test_metrics_default_thresh.json` | Verified ResNet-18 extended metrics |
| 6 | `reports/gradcam/` | GradCAM++ figures (use as paper figures) |
| 7 | `kaggle_train.py` | Actual training code for ConvNeXt + CXformer |
| 8 | `src/adaptive_fusion.py` | 4-modal architecture (understand what was NOT trained) |
| 9 | `src/knowledge_graph.py` | Understand the rule-based verifier |

---

## 8. Suggested Order for Writing the Paper

1. **Abstract** — write last; 150 words; include ROC-AUC 0.988, recall 99.5%, 2-modal hybrid, paediatric cohort, explainability, not clinical
2. **Introduction** — problem, why explainability matters, our contributions (3 verified + 2 proposed)
3. **Related Work** — CheXNet, DINOv2/RAD-DINO, CXformer, R2Gen, GradCAM++
4. **Dataset** — Kermany paediatric, split strategy (85/15 stratified, seed=42), class imbalance, test set policy
5. **Individual Models** — ResNet-18 (baseline), ConvNeXt-Tiny, CXformer-S (focus on architecture differences, training settings)
6. **Hybrid Fusion** — feature-level design, why frozen encoders, 0.428M trained parameters; include architecture figure
7. **Explainability** — GradCAM++ setup, heatmap examples, failure mode analysis; caveat no quantitative eval
8. **Evidence Verification** — describe rule-based verifier as proposed/architectural contribution; do not report classification results for it
9. **Multi-Agent Reasoning** — describe as an architectural contribution; do not report evaluation
10. **Experiments** — dataset, split, hardware, training settings, evaluation procedure
11. **Results** — Table comparing all models; focus on verified numbers; add caveat for CXformer/Kaggle ConvNeXt
12. **Discussion** — recall vs specificity trade-off, hybrid effectiveness, distribution shift observation, limitations
13. **Conclusion** — summary of verified contributions, future work (train 4-modal, quantitative explainability eval, adult cohort)
14. **Abstract** — write now

---

## 9. Before Submitting the Paper — Checklist

- [ ] Download and verify CXformer-S test_metrics.json from Kaggle (CM cross-check)
- [ ] Download and verify Kaggle T4 ConvNeXt-Tiny test_metrics.json from Kaggle
- [ ] Download training history CSVs for ConvNeXt (8ep) and CXformer (10ep) for training curves
- [ ] Retrieve hybrid training script/notebook and document exact hyperparameters
- [ ] Compute precision for CXformer-S and Kaggle ConvNeXt-Tiny from their confusion matrices
- [ ] Confirm CXformer-S paper citation (MIDL 2025, Al-Mahrooqi et al.)
- [ ] Confirm IU X-ray / R2Gen citation details
- [ ] Confirm EigenCAM citation
- [ ] Generate ROC curve plot for all models (requires re-running evaluation with prob outputs)
- [ ] Do NOT claim the 4-modal AdaptiveFusionModel achieved any classification results
- [ ] Do NOT call the rule-based verifier a "knowledge graph"
- [ ] Add clinical disclaimer: "not for clinical diagnosis or treatment decisions"
