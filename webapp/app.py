"""
webapp/app.py
=============
MedReason++ Showcase Web Application — FastAPI Backend

Endpoints:
  GET  /                    → serves the main HTML page
  POST /analyse             → analyse an uploaded X-ray image
  GET  /gradcam/{filename}  → serve a GradCAM PNG from reports/gradcam/
  GET  /results             → last analysis result (JSON)

Run:
    cd webapp
    uvicorn app:app --reload --port 8000

Then open: http://localhost:8000
"""
import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Add src/ to path ──────────────────────────────────────────────────
WEBAPP_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Load stored API key (never sent to browser) ───────────────────────
from config import GEMINI_API_KEY, GEMINI_MODEL
WEBAPP_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

app = FastAPI(title="MedReason++", version="1.0")

# ── Static files ──────────────────────────────────────────────────────
app.mount("/static",  StaticFiles(directory=str(WEBAPP_DIR / "static")),       name="static")
app.mount("/gradcam", StaticFiles(directory=str(PROJECT_ROOT / "reports" / "gradcam")), name="gradcam")

# ── Pre-load models at startup so first request is instant ────────────
_extractor = None

@app.on_event("startup")
async def preload_models():
    """Load all heavy models at server startup, not on first request."""
    global _extractor
    print("\n[Startup] Pre-loading models — this takes 1–2 min on CPU...")
    try:
        from extract_features import FeatureExtractor
        _extractor = FeatureExtractor()
        print("[Startup] All models loaded. Server ready.\n")
    except Exception as e:
        print(f"[Startup] Warning: model pre-load failed: {e}")
        print("[Startup] Models will load on first request instead.\n")

def get_extractor():
    global _extractor
    if _extractor is None:
        from extract_features import FeatureExtractor
        _extractor = FeatureExtractor()
    return _extractor


# ── SHAP explainer (lazy-loaded separately) ───────────────────────────
_shap_explainer = None

def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        from shap_explainer import SHAPExplainer
        _shap_explainer = SHAPExplainer()
    return _shap_explainer


# ── Helper ────────────────────────────────────────────────────────────

def _format_response(ctx: dict) -> dict:
    """Extract API-safe fields from a full pipeline context dict."""
    return {
        "prediction":         ctx.get("prediction", "UNKNOWN"),
        "confidence":         ctx.get("confidence", 0.0),
        "prob_pneumonia":     ctx.get("prob_pneumonia", ctx.get("prob_pn", 0.0)),
        "prob_normal":        ctx.get("prob_normal", 0.0),
        "activation_level":   ctx.get("activation_level", 0.0),
        "activation_region":  ctx.get("activation_region", "unknown"),
        "gradcam_heatmap":    ctx.get("gradcam_heatmap_b64", ""),
        "gradcam_overlay":    ctx.get("gradcam_overlay_b64", ""),
        # Multimodal fusion
        "fusion_prediction":  ctx.get("fusion_prediction", ""),
        "fusion_prob_pn":     ctx.get("fusion_prob_pn", 0.0),
        "modality_weights":   ctx.get("modality_weights", {}),
        # Knowledge graph
        "kg_score":           ctx.get("kg_score", 0.0),
        "kg_consistency":     ctx.get("kg_consistency", ""),
        "kg_supporting":      ctx.get("kg_supporting", []),
        "kg_contradicting":   ctx.get("kg_contradicting", []),
        # Conflict
        "conflict_detected":  ctx.get("conflict_detected", False),
        "conflict_severity":  ctx.get("conflict_severity", "NONE"),
        "conflict_list":      ctx.get("conflict_list", []),
        # Agents
        "agent_report":       ctx.get("final_report", None),
        "agent_findings":     [
            {"agent": f["agent"], "finding": f["finding"]}
            for f in ctx.get("agent_findings", [])
            if f["agent"] != "evidence_aggregator"
        ],
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = WEBAPP_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "project": "MedReason++"}


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    report_text: str = Form(default=""),
    metadata_age: str = Form(default=""),
    metadata_symptoms: str = Form(default="0"),
    metadata_followup: str = Form(default="0"),
):
    """Analyse an uploaded chest X-ray with full pipeline (API key from server config)."""
    if not file.filename.lower().endswith((".jpeg", ".jpg", ".png")):
        raise HTTPException(400, "Please upload a JPEG or PNG chest X-ray image.")

    meta = {
        "has_symptoms": metadata_symptoms == "1",
        "is_followup":  metadata_followup == "1",
    }
    if metadata_age:
        try: meta["age_years"] = float(metadata_age)
        except: pass

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extractor = get_extractor()
        ctx = extractor.extract(tmp_path)

        if report_text and report_text.strip():
            try:
                from knowledge_graph import MedicalKnowledgeGraph
                from adaptive_fusion import ConflictDetector, AdaptiveFusionModel
                from text_encoder import ReportTextEncoder, MetadataEncoder
                import torch

                kg = MedicalKnowledgeGraph()
                kg_result = kg.verify(report_text.strip(), ctx["prediction"], ctx["confidence"])
                ctx["kg_score"]       = kg_result.kg_score
                ctx["kg_consistency"] = kg_result.consistency
                ctx["kg_supporting"]  = kg_result.supporting_findings
                ctx["kg_contradicting"]= kg_result.contradicting_findings
                ctx["kg_formatted"]   = kg.format_for_agent(kg_result)

                text_enc  = ReportTextEncoder()
                text_vec  = torch.from_numpy(text_enc.encode(report_text.strip())).unsqueeze(0)
                meta = {"age_years": float(metadata_age) if metadata_age else None,
                        "has_symptoms": metadata_symptoms == "1"}
                meta_vec  = MetadataEncoder.build_vector(**{k:v for k,v in meta.items() if v is not None})
                meta_mod  = MetadataEncoder(proj_dim=64)
                meta_feat = meta_mod(meta_vec)
                fusion    = AdaptiveFusionModel().eval()
                cx_feat   = torch.tensor([ctx["convnext_features"]], dtype=torch.float32)
                cxf_feat  = torch.tensor([ctx["cxformer_features"]], dtype=torch.float32)
                with torch.no_grad():
                    _, attn = fusion(cx_feat, cxf_feat, text_vec, meta_feat)
                ctx["modality_weights"] = fusion.get_modality_contributions(attn)

                cd = ConflictDetector()
                conflict = cd.detect(
                    prediction=ctx["prediction"], confidence=ctx["confidence"],
                    attn_weights=ctx["modality_weights"],
                    kg_score=kg_result.kg_score, kg_consistency=kg_result.consistency,
                    text_supports_pn=kg_result.kg_score > 0.1,
                    image_supports_pn=ctx["prediction"] == "PNEUMONIA",
                )
                ctx["conflict_detected"] = conflict["conflicts_found"]
                ctx["conflict_severity"] = conflict["severity"]
                ctx["conflict_list"]     = conflict["conflict_list"]
            except Exception as e:
                print(f"[KG/Fusion error] {e}")

        if GEMINI_API_KEY:
            try:
                from medreason_agents import (GeminiClient, agent_visual_analyst,
                    agent_clinical_reasoner, agent_evidence_aggregator_full)
                client = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
                ctx = agent_visual_analyst(ctx, client)
                ctx = agent_clinical_reasoner(ctx, client)
                ctx = agent_evidence_aggregator_full(ctx, client)
            except Exception as e:
                print(f"[Agent error] {e}")
                ctx.setdefault("agent_findings", [])
                ctx["final_report"] = (
                    "Multi-agent reasoning temporarily unavailable. "
                    "Image classification, GradCAM++, knowledge graph, and conflict detection results are complete."
                )

        return JSONResponse(_format_response(ctx))
    finally:
        os.unlink(tmp_path)


@app.get("/sample-images")
async def sample_images():
    """Return a list of test images available for demo."""
    test_dir = PROJECT_ROOT / "archive (7)" / "chest_xray" / "test"
    samples = {"NORMAL": [], "PNEUMONIA": []}
    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = test_dir / cls
        if folder.is_dir():
            files = sorted(folder.glob("*.jpeg"))[:3]
            samples[cls] = [str(f) for f in files]
    return JSONResponse(samples)


@app.post("/analyse-sample")
async def analyse_sample(body: dict):
    """Analyse a built-in sample image (API key from server config)."""
    image_path  = body.get("image_path", "")
    report_text = body.get("report_text", "")
    metadata    = body.get("metadata", {})

    p = Path(image_path)
    if not p.exists() or p.suffix.lower() not in (".jpeg", ".jpg", ".png"):
        raise HTTPException(400, "Invalid sample image path.")

    # Always extract image features first — never fails
    extractor = get_extractor()
    ctx = extractor.extract(str(p))

    # Add KG verification if report text provided
    if report_text and report_text.strip():
        try:
            from knowledge_graph import MedicalKnowledgeGraph
            from adaptive_fusion import ConflictDetector
            from text_encoder import MetadataEncoder
            import torch

            kg = MedicalKnowledgeGraph()
            kg_result = kg.verify(report_text.strip(), ctx["prediction"], ctx["confidence"])
            ctx["kg_score"]         = kg_result.kg_score
            ctx["kg_consistency"]   = kg_result.consistency
            ctx["kg_supporting"]    = kg_result.supporting_findings
            ctx["kg_contradicting"] = kg_result.contradicting_findings
            ctx["kg_conflict"]      = kg_result.conflict_detected
            ctx["kg_formatted"]     = kg.format_for_agent(kg_result)

            # Add text + metadata fusion weights
            from text_encoder import ReportTextEncoder, TextProjectionHead
            text_enc = ReportTextEncoder()
            text_vec = torch.from_numpy(text_enc.encode(report_text.strip())).unsqueeze(0)
            meta_vec = MetadataEncoder.build_vector(
                age_years=metadata.get("age_years") if metadata else None,
                has_symptoms=metadata.get("has_symptoms", False) if metadata else False,
            )
            from adaptive_fusion import AdaptiveFusionModel
            import torch.nn as nn
            fusion = AdaptiveFusionModel().eval()
            cx_feat   = torch.tensor([ctx["convnext_features"]], dtype=torch.float32)
            cxf_feat  = torch.tensor([ctx["cxformer_features"]], dtype=torch.float32)
            meta_enc  = MetadataEncoder(proj_dim=64)
            meta_feat = meta_enc(meta_vec)
            with torch.no_grad():
                _, attn = fusion(cx_feat, cxf_feat, text_vec, meta_feat)
            ctx["modality_weights"] = fusion.get_modality_contributions(attn)

            cd = ConflictDetector()
            conflict = cd.detect(
                prediction=ctx["prediction"], confidence=ctx["confidence"],
                attn_weights=ctx["modality_weights"],
                kg_score=kg_result.kg_score, kg_consistency=kg_result.consistency,
                text_supports_pn=kg_result.kg_score > 0.1,
                image_supports_pn=ctx["prediction"] == "PNEUMONIA",
            )
            ctx["conflict_detected"] = conflict["conflicts_found"]
            ctx["conflict_severity"] = conflict["severity"]
            ctx["conflict_list"]     = conflict["conflict_list"]
        except Exception as e:
            print(f"[KG/Fusion error] {e}")

    # Run Gemini agents if API key is set — gracefully skip on quota/error
    if GEMINI_API_KEY:
        try:
            from medreason_agents import (GeminiClient, agent_visual_analyst,
                agent_clinical_reasoner, agent_evidence_aggregator_full)
            client = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
            ctx = agent_visual_analyst(ctx, client)
            ctx = agent_clinical_reasoner(ctx, client)
            ctx = agent_evidence_aggregator_full(ctx, client)
        except Exception as e:
            print(f"[Agent error] {e}")
            ctx.setdefault("agent_findings", [])
            ctx["final_report"] = (
                "Multi-agent reasoning is temporarily unavailable. "
                "Image classification, GradCAM++, knowledge graph verification, "
                "and conflict detection results above are complete."
            )

    return JSONResponse(_format_response(ctx))
