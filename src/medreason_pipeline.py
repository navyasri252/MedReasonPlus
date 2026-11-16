"""
src/medreason_pipeline.py
==========================
Complete MedReason++ Inference Pipeline

Integrates all components:
  1. Image feature extraction (ConvNeXt + CXformer)
  2. GradCAM++ explainability
  3. Report text encoding (sentence-transformers)
  4. Metadata encoding
  5. Adaptive multimodal fusion with attention gating
  6. Medical knowledge graph verification
  7. Conflict detection
  8. Gemini multi-agent reasoning (3 agents)

This is the single entry point for the web app and standalone use.

Usage:
    pipeline = MedReasonPipeline(gemini_api_key="...")
    result = pipeline.run(
        image_path="path/to/xray.jpeg",
        report_text="The lungs are clear...",  # optional
        metadata={"age_years": 8, "has_symptoms": True},  # optional
    )
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class MedReasonPipeline:
    """
    Full MedReason++ inference pipeline.

    Loads all models once and reuses them for multiple inferences.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model:   str = "auto",
        device:         Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"\n[MedReasonPipeline] Initialising on {self.device}…")

        # Lazy imports to keep startup time reasonable
        self._extractor    = None
        self._text_encoder = None
        self._kg           = None
        self._gemini       = None
        self._gemini_key   = gemini_api_key
        self._gemini_model = gemini_model
        self._fusion_model = None

        print("[MedReasonPipeline] Ready. Models load on first call.")

    # ── Lazy loaders ──────────────────────────────────────────────────

    def _get_extractor(self):
        if self._extractor is None:
            from extract_features import FeatureExtractor
            self._extractor = FeatureExtractor(device=str(self.device))
        return self._extractor

    def _get_text_encoder(self):
        if self._text_encoder is None:
            from text_encoder import ReportTextEncoder
            self._text_encoder = ReportTextEncoder(device=str(self.device))
        return self._text_encoder

    def _get_kg(self):
        if self._kg is None:
            from knowledge_graph import MedicalKnowledgeGraph
            self._kg = MedicalKnowledgeGraph()
        return self._kg

    def _get_gemini(self):
        if self._gemini is None and self._gemini_key:
            from medreason_agents import GeminiClient
            self._gemini = GeminiClient(
                api_key=self._gemini_key, model=self._gemini_model)
        return self._gemini

    def _get_fusion_model(self):
        """Build and return the adaptive fusion model (untrained = random head)."""
        if self._fusion_model is None:
            from adaptive_fusion import AdaptiveFusionModel
            self._fusion_model = AdaptiveFusionModel().to(self.device).eval()
        return self._fusion_model

    # ── Main run ──────────────────────────────────────────────────────

    def run(
        self,
        image_path:   str,
        report_text:  Optional[str] = None,
        metadata:     Optional[dict] = None,
        run_agents:   bool = True,
        save_output:  bool = True,
    ) -> dict:
        """
        Run the full pipeline on one chest X-ray.

        Parameters
        ----------
        image_path  : path to .jpeg chest X-ray
        report_text : optional radiology report text
        metadata    : optional dict with keys:
                        age_years, has_symptoms, is_followup,
                        study_image_count, report_word_count
        run_agents  : whether to call Gemini agents (requires API key)
        save_output : whether to save JSON to reports/agent_reports/

        Returns
        -------
        dict — complete shared context
        """
        t_start = time.time()
        print(f"\n{'='*60}")
        print("  MedReason++ Full Pipeline")
        print(f"  Image: {Path(image_path).name}")
        print(f"{'='*60}")

        ctx: dict = {
            "image_path":           str(image_path),
            "report_text":          report_text or "",
            "metadata":             metadata or {},
            "agent_findings":       [],
            "final_report":         "",
        }

        # ── Step 1: Image features + GradCAM ─────────────────────────
        print("\n[1/6] Extracting image features and GradCAM++…")
        extractor = self._get_extractor()
        img_ctx = extractor.extract(image_path)
        ctx.update({k: v for k, v in img_ctx.items()
                    if k not in ("agent_findings", "final_report")})
        print(f"  Image prediction: {ctx['prediction']} ({ctx['confidence']*100:.1f}%)")
        print(f"  Activation: {ctx['activation_region']}")

        # ── Step 2: Text encoding ─────────────────────────────────────
        print("\n[2/6] Encoding report text…")
        if report_text and report_text.strip():
            text_enc = self._get_text_encoder()
            text_feat = text_enc.encode_tensor(report_text)  # [1, 384]
            ctx["text_features"]  = text_feat[0].tolist()
            ctx["report_encoded"] = True
            print(f"  Text encoded: {len(report_text.split())} words → 384-dim vector")
        else:
            ctx["text_features"]  = [0.0] * 384
            ctx["report_encoded"] = False
            print("  No report text provided — using zero vector")

        # ── Step 3: Metadata encoding ─────────────────────────────────
        print("\n[3/6] Encoding metadata…")
        from text_encoder import MetadataEncoder
        meta_vec = MetadataEncoder.build_vector(
            age_years=         metadata.get("age_years")          if metadata else None,
            has_symptoms=      metadata.get("has_symptoms", False) if metadata else False,
            is_followup=       metadata.get("is_followup",  False) if metadata else False,
            study_image_count= metadata.get("study_image_count", 2) if metadata else 2,
            report_word_count= len(report_text.split()) if report_text else 30,
        )
        meta_enc  = MetadataEncoder(proj_dim=64)
        meta_feat = meta_enc(meta_vec)   # [1, 64]
        ctx["metadata_features"] = meta_vec[0].tolist()
        print(f"  Metadata vector: {meta_vec[0].tolist()}")

        # ── Step 4: Adaptive fusion ───────────────────────────────────
        print("\n[4/6] Running adaptive multimodal fusion…")
        fusion = self._get_fusion_model()

        cx_feat   = torch.tensor([ctx["convnext_features"]],  dtype=torch.float32)
        cxf_feat  = torch.tensor([ctx["cxformer_features"]],  dtype=torch.float32)
        text_feat = torch.tensor([ctx["text_features"]],       dtype=torch.float32)

        with torch.no_grad():
            fusion_logits, attn_weights = fusion(cx_feat, cxf_feat, text_feat, meta_feat)
            fusion_probs = torch.softmax(fusion_logits, dim=1)[0]

        modality_contribs = fusion.get_modality_contributions(attn_weights)
        ctx["fusion_prediction"]   = "PNEUMONIA" if fusion_probs[1] > 0.5 else "NORMAL"
        ctx["fusion_prob_pn"]      = round(float(fusion_probs[1].item()), 4)
        ctx["fusion_prob_normal"]  = round(float(fusion_probs[0].item()), 4)
        ctx["modality_weights"]    = modality_contribs
        ctx["attention_weights"]   = attn_weights[0].tolist()

        print(f"  Fusion prediction: {ctx['fusion_prediction']} "
              f"({ctx['fusion_prob_pn']*100:.1f}%)")
        print("  Modality contributions:")
        for m, w in modality_contribs.items():
            print(f"    {m}: {w}%")

        # ── Step 5: Knowledge graph + conflict detection ──────────────
        print("\n[5/6] Knowledge graph verification + conflict detection…")
        kg = self._get_kg()
        kg_result = kg.verify(
            report_text=report_text or "",
            prediction=ctx["prediction"],
            confidence=ctx["confidence"],
        )
        ctx["kg_score"]          = kg_result.kg_score
        ctx["kg_consistency"]    = kg_result.consistency
        ctx["kg_supporting"]     = kg_result.supporting_findings
        ctx["kg_contradicting"]  = kg_result.contradicting_findings
        ctx["kg_conflict"]       = kg_result.conflict_detected
        ctx["kg_conflict_reason"]= kg_result.conflict_reason or ""
        ctx["kg_modifier"]       = kg_result.recommendation_modifier
        ctx["kg_formatted"]      = kg.format_for_agent(kg_result)

        from adaptive_fusion import ConflictDetector
        cd = ConflictDetector()
        image_pn  = ctx["prediction"] == "PNEUMONIA"
        text_pn   = kg_result.kg_score > 0.1 if report_text else image_pn

        conflict_result = cd.detect(
            prediction=ctx["prediction"],
            confidence=ctx["confidence"],
            attn_weights=modality_contribs,
            kg_score=kg_result.kg_score,
            kg_consistency=kg_result.consistency,
            text_supports_pn=text_pn,
            image_supports_pn=image_pn,
        )
        ctx["conflict_detected"] = conflict_result["conflicts_found"]
        ctx["conflict_severity"] = conflict_result["severity"]
        ctx["conflict_list"]     = conflict_result["conflict_list"]
        ctx["conflict_explanation"] = conflict_result["explanation"]

        print(f"  KG score: {kg_result.kg_score:.3f}  Consistency: {kg_result.consistency}")
        print(f"  Conflict severity: {conflict_result['severity']}")
        if conflict_result["conflict_list"]:
            for c in conflict_result["conflict_list"]:
                print(f"    ⚠ {c}")

        # ── Step 6: Multi-agent reasoning ─────────────────────────────
        if run_agents:
            gemini = self._get_gemini()
            if gemini:
                print("\n[6/6] Running multi-agent Gemini reasoning…")
                from medreason_agents import (
                    agent_visual_analyst,
                    agent_clinical_reasoner,
                    agent_evidence_aggregator_full,
                )
                ctx = agent_visual_analyst(ctx, gemini)
                ctx = agent_clinical_reasoner(ctx, gemini)
                ctx = agent_evidence_aggregator_full(ctx, gemini)
            else:
                print("\n[6/6] Skipping agents — no Gemini API key provided.")
        else:
            print("\n[6/6] Agents skipped (run_agents=False).")

        ctx["total_time_s"] = round(time.time() - t_start, 1)

        # ── Save output ───────────────────────────────────────────────
        if save_output:
            out_dir = _SRC.parent / "reports" / "agent_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            img_stem = Path(image_path).stem
            out_path = out_dir / f"{img_stem}_full_report.json"

            safe_ctx = {k: v for k, v in ctx.items()
                        if k not in ("convnext_features", "cxformer_features",
                                     "text_features", "metadata_features",
                                     "gradcam_heatmap_b64", "gradcam_overlay_b64")}
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(safe_ctx, f, indent=2)
            print(f"\n  Report saved: {out_path}")

        print(f"\n  Total time: {ctx['total_time_s']}s")
        print("=" * 60)
        return ctx
