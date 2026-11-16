# Claim Verification Table — MedReason++

Each major contribution and numerical result is listed with its verification status.

**Status definitions:**
- **Verified from code/output** — confirmed by reading the actual code + cross-checking JSON metrics against confusion matrices
- **Documented but not independently verified** — appears in `experiment_results.md` but no local JSON to cross-check
- **Implemented but not evaluated** — code exists and runs, but no quantitative evaluation on a held-out dataset
- **Proposed / demo-only** — architecture defined, never trained, outputs are from random weights
- **Unsupported / do not claim** — not supported by any evidence in the project

---

| # | Claim | Evidence source | Status | Safe wording for the paper | Additional verification needed |
|---|-------|----------------|--------|---------------------------|-------------------------------|
| 1 | ResNet-18 baseline test accuracy = 83.81% | `processed/phase6/test_metrics.json`; CM cross-check passes | **Verified from code/output** | "ResNet-18 baseline achieved 83.81% accuracy on the 624-image test set" | None |
| 2 | ResNet-18 baseline recall = 99.74% | Same JSON | **Verified from code/output** | "ResNet-18 baseline achieved 99.7% sensitivity" | None |
| 3 | ResNet-18 baseline ROC-AUC = 0.961 | Same JSON | **Verified from code/output** | "ROC-AUC of 0.961" | None |
| 4 | ResNet-18 baseline CM: TN=134, FP=100, FN=1, TP=389 | Same JSON | **Verified from code/output** | State as-is | None |
| 5 | ResNet-18 extended accuracy = 88.78% | `processed/phase6b/test_metrics_default_thresh.json`; CM cross-check passes | **Verified from code/output** | "Extended ResNet-18 achieved 88.8% accuracy" | None |
| 6 | ResNet-18 extended recall = 99.23% | Same JSON | **Verified from code/output** | "Sensitivity 99.2%" | None |
| 7 | ResNet-18 extended ROC-AUC = 0.9679 | Same JSON | **Verified from code/output** | "ROC-AUC 0.968" | None |
| 8 | ResNet-18 extended CM: TN=167, FP=67, FN=3, TP=387 | Same JSON | **Verified from code/output** | State as-is | None |
| 9 | Hybrid 2-modal accuracy = 90.71% | `processed/hybrid_convnext_cxformer/test_metrics.json`; CM cross-check passes | **Verified from code/output** | "The hybrid model achieved 90.7% accuracy on the 624-image test set" | None |
| 10 | Hybrid 2-modal recall = 99.49% | Same JSON | **Verified from code/output** | "Sensitivity 99.5%; only 2 of 390 pneumonia cases missed" | None |
| 11 | Hybrid 2-modal ROC-AUC = 0.9876 | Same JSON | **Verified from code/output** | "ROC-AUC 0.988, the highest of all evaluated models" | None |
| 12 | Hybrid 2-modal F1 = 0.9305 | Same JSON | **Verified from code/output** | "F1-score 0.930" | None |
| 13 | Hybrid 2-modal CM: TN=178, FP=56, FN=2, TP=388 | Same JSON | **Verified from code/output** | State as-is | None |
| 14 | ConvNeXt-Tiny test accuracy = 0.867 (local run) | `processed/convnext/test_metrics.json`; CM cross-check passes | **Verified from code/output** | "A local CPU-trained ConvNeXt-Tiny run achieved 86.7% accuracy" | Distinguish from Kaggle 8-epoch run |
| 15 | ConvNeXt-Tiny (Kaggle T4, 8ep) accuracy = 0.8574 | `reports/experiment_results.md` | **Documented but not independently verified** | "ConvNeXt-Tiny trained on T4 GPU achieved 85.7% accuracy (from training log)" | Download original Kaggle test_metrics.json and verify CM |
| 16 | ConvNeXt-Tiny (Kaggle T4, 8ep) ROC-AUC = 0.9713 | `reports/experiment_results.md` | **Documented but not independently verified** | "ROC-AUC 0.971 (from training log)" | Same as above |
| 17 | ConvNeXt-Tiny (Kaggle T4, 8ep) CM: TN=147, FP=87, FN=2, TP=388 | `reports/experiment_results.md` | **Documented but not independently verified** | Conditionally state with verification note | Verify from Kaggle output |
| 18 | CXformer-S accuracy = 0.9103 | `reports/experiment_results.md` | **Documented but not independently verified** | "CXformer-S achieved 91.0% accuracy (from training log)" | Download Kaggle cxformer/test_metrics.json and verify |
| 19 | CXformer-S recall = 0.9590 | `reports/experiment_results.md` | **Documented but not independently verified** | "Sensitivity 95.9%" | Same as above |
| 20 | CXformer-S specificity = 0.8291 | `reports/experiment_results.md` | **Documented but not independently verified** | "Specificity 82.9%" | Same as above |
| 21 | CXformer-S ROC-AUC = 0.9702 | `reports/experiment_results.md` | **Documented but not independently verified** | "ROC-AUC 0.970" | Same as above |
| 22 | CXformer-S CM: TN=194, FP=40, FN=16, TP=374 | `reports/experiment_results.md` | **Documented but not independently verified** | State conditionally | Verify from Kaggle output |
| 23 | Hybrid outperforms individual models on ROC-AUC | Local hybrid JSON (0.9876) vs documented ConvNeXt (0.9713) and CXformer (0.9702) | **Partially verified** — hybrid verified locally; others from docs | "Feature-level fusion of ConvNeXt-Tiny and CXformer-S achieved ROC-AUC 0.988, compared to 0.971 and 0.970 for each model independently" | Verify individual model metrics from Kaggle |
| 24 | Hybrid achieves near-perfect recall (only 2 FN) | `processed/hybrid_convnext_cxformer/test_metrics.json`, FN=2 | **Verified from code/output** | "The hybrid model missed only 2 of 390 pneumonia cases" | None |
| 25 | All experiments use the same 624-image test set, never touched during training | `kaggle_train.py` code — test set collected from original test folder, never added to train/val | **Verified from code/output** | "All models evaluated on a held-out 624-image test set (234 NORMAL, 390 PNEUMONIA), not used during training or validation" | None |
| 26 | GradCAM++ activates on lung parenchyma for true positives | `reports/gradcam/convnext_true_positive.png` — visual inspection | **Implemented but not evaluated** | "Visual inspection of GradCAM++ heatmaps suggests the model activates over lung parenchyma regions for correctly classified pneumonia cases" | Radiologist annotation comparison needed for strong claim |
| 27 | GradCAM++ false positive activation on non-lung structures | `reports/gradcam/convnext_false_positive.png` — visual inspection | **Implemented but not evaluated** | "False positive heatmaps suggest activation over non-pathological structures such as cardiac shadow and ribs" | Same |
| 28 | Adaptive 4-modality fusion with attention gate | `src/adaptive_fusion.py` — architecture implemented | **Proposed / demo-only** | "We propose an adaptive 4-modality fusion architecture with a learnable attention gate; this architecture has not yet been trained" | Train on Kermany with text labels before claiming results |
| 29 | Attention gate shows which modality is trusted most | Pipeline output — but from untrained random weights | **Unsupported / do not claim** | Do NOT claim attention weights are meaningful | Train AdaptiveFusionModel before any claims |
| 30 | Rule-based evidence verification (called "knowledge graph") | `src/knowledge_graph.py` — 12 regex-based Finding rules in a list | **Implemented but not evaluated** | "A rule-based evidence verification system using 12 hand-coded clinical patterns extracts a pneumonia evidence score from report text" | Evaluate on held-out radiology report corpus with ground truth |
| 31 | Conflict detection identifies image-text disagreements | `src/adaptive_fusion.py::ConflictDetector` — functional tests pass | **Implemented but not evaluated** | "A conflict detection module flags disagreements between the image classifier prediction and the rule-based text evidence" | Evaluate precision/recall of conflict detection vs human annotation |
| 32 | Gemini agents produce structured clinical reports | System runs and produces output when quota allows | **Implemented but not evaluated** | "A three-agent Gemini-based reasoning chain produces structured reports; these were not evaluated for clinical accuracy" | Radiologist evaluation study needed |
| 33 | Multi-agent system uses "shared memory" | `ctx` dict passed between three sequential function calls | **Verified from code/output** | "Three agents share a common context dictionary containing prediction scores, heatmap activation data, and evidence verification results" | None — but clarify it is a Python dict, not a database |
| 34 | System beats Kaggle baseline (~77%) | Kaggle baseline value is informal community reference; all 5 models exceed it | **Documented but not independently verified** | "All evaluated models exceed the widely-cited Kaggle community benchmark of approximately 77% accuracy" | Cite specific Kaggle notebook if available |
| 35 | Results are on paediatric chest X-rays only | Kermany dataset documentation; confirmed by project inspection | **Verified from code/output** | "All experiments were conducted on the Kermany paediatric chest X-ray dataset; results may not generalise to adult populations" | None |
| 36 | CXformer-S trained with frozen backbone (linear probing) | `kaggle_train.py` line: `for p in cxf_backbone.parameters(): p.requires_grad = False` | **Verified from code/output** | "CXformer-S backbone was frozen; only the 384→2 classification head was trained (linear probing)" | None |
| 37 | Hybrid 2-modal model has 0.428M trainable parameters | `experiment_results.md` — trainable params documented | **Documented but not independently verified** | "The hybrid model trained 0.428M parameters (projection layers and classifier head) with 49.9M backbone parameters frozen" | Recompute from checkpoint parameter count |
| 38 | ConvNeXt-Tiny full fine-tuning (all 27.8M params) | `kaggle_train.py` — all parameters in optimizer | **Verified from code/output** | "ConvNeXt-Tiny was fully fine-tuned from ImageNet pretrained weights" | None |
| 39 | Text encoder uses all-MiniLM-L6-v2 (384-dim) | `src/text_encoder.py::MODEL_NAME = "all-MiniLM-L6-v2"` | **Verified from code/output** | "Report text was encoded using the all-MiniLM-L6-v2 sentence transformer model (384-dim)" | None |
| 40 | AdaptiveFusionModel was never trained | `src/medreason_pipeline.py::_get_fusion_model()` — no `load_state_dict`, no checkpoint | **Verified from code/output** | Do NOT attribute classification results to the 4-modal model | None needed — this is a confirmed limitation |
