"""
src/medreason_agents.py
========================
MedReason++ Multi-Agent Clinical Reasoning System.

Three Gemini agents share a common context (SharedContext) and
produce an evidence-grounded clinical report:

  Agent 1 — Visual Analyst
      Reads:  GradCAM heatmap description + prediction
      Writes: Natural language description of visual evidence

  Agent 2 — Clinical Reasoner
      Reads:  Agent 1 finding + CXformer feature summary + confidence
      Writes: Evidence-grounded reasoning paragraph

  Agent 3 — Evidence Aggregator
      Reads:  All previous findings
      Writes: Structured final recommendation

⚠ DISCLAIMER: For research purposes only. NOT for clinical use.

Usage:
    Set GEMINI_API_KEY environment variable, then:
    python src/medreason_agents.py --image path/to/xray.jpeg
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Gemini client wrapper
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the Google GenAI SDK (google-genai)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "auto"):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai not installed.\n"
                "Run: pip install google-genai"
            )
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set.\n"
                "Set it with: export GEMINI_API_KEY=your_key\n"
                "or pass api_key= to GeminiClient()"
            )
        self._client = genai.Client(api_key=key)

        # Auto-detect working model if not specified
        if model == "auto":
            model = self._find_working_model()
        self._model = model
        print(f"[GeminiClient] Using model: {model}")

    def _find_working_model(self) -> str:
        """Try candidate models. Never raises — returns first candidate on failure."""
        candidates = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview-05-20",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
        # Quick single probe — don't loop through all if first one gives quota error
        for m in candidates:
            try:
                r = self._client.models.generate_content(
                    model=m, contents="Say OK")
                if r.text:
                    print(f"[GeminiClient] Model: {m}")
                    return m
            except Exception as exc:
                err = str(exc)
                if "429" in err or "quota" in err.lower():
                    # Quota exhausted — no point trying other models
                    print(f"[GeminiClient] Quota exhausted on {m}. Skipping remaining models.")
                    return m   # ask() will return clean message
                # 404 not found — try next
                continue
        print(f"[GeminiClient] Defaulting to {candidates[0]}.")
        return candidates[0]

    def ask(self, prompt: str, max_retries: int = 3) -> str:
        """Send a prompt and return the text response."""
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=prompt)
                return response.text.strip()
            except Exception as exc:
                err_str = str(exc)
                # Parse quota exhaustion specifically
                if "429" in err_str and ("RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()):
                    # Extract retry time if available
                    import re
                    retry_match = re.search(r'retry in ([\d]+h[\d.]+s|[\d.]+s)', err_str)
                    retry_info = f" Retry available in: {retry_match.group(1)}." if retry_match else ""
                    return (
                        f"Daily API quota reached for the Gemini free tier (limit: 20 requests/day).{retry_info} "
                        f"The image analysis, GradCAM, and knowledge graph results above are complete and do not require the API. "
                        f"The multi-agent reasoning report will be available after the quota resets."
                    )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"Reasoning service temporarily unavailable. Image analysis and knowledge graph results remain valid."
        return "Reasoning service unavailable."


# ---------------------------------------------------------------------------
# Agent 1 — Visual Analyst
# ---------------------------------------------------------------------------

def agent_visual_analyst(ctx: dict, client: GeminiClient) -> dict:
    """
    Analyse the GradCAM activation pattern and describe what it means
    visually in clinical language.
    """
    prompt = f"""You are a radiology AI assistant helping to explain a deep learning model's decision.

A chest X-ray has been analysed by a convolutional neural network (ConvNeXt-Tiny).

Model prediction: {ctx['prediction']}
Confidence: {round(ctx['confidence']*100, 1)}%
Primary activation region: {ctx['activation_region']}
Activation intensity (0=low, 1=high): {ctx['activation_level']:.3f}

The GradCAM++ heatmap shows that the model activated most strongly in the {ctx['activation_region']}.

Task: In 2-3 concise clinical sentences, describe:
1. What this activation pattern suggests anatomically
2. Whether the activation region is consistent with typical pneumonia presentation
3. Any caveats about interpreting this finding

Important:
- This is a RESEARCH tool, not a clinical diagnosis
- Use professional but accessible language
- Do NOT make a definitive diagnosis
- Focus on what the visual evidence suggests

Respond with only the 2-3 sentences, no headers."""

    finding = client.ask(prompt)
    ctx["agent_findings"].append({"agent": "visual_analyst", "finding": finding})
    print(f"[Agent 1 — Visual Analyst]\n{finding}\n")
    return ctx


# ---------------------------------------------------------------------------
# Agent 2 — Clinical Reasoner
# ---------------------------------------------------------------------------

def agent_clinical_reasoner(ctx: dict, client: GeminiClient) -> dict:
    """
    Given the visual finding, KG verification, and CXformer feature summary,
    reason about supporting and contradicting evidence.
    """
    visual_finding = next(
        (f["finding"] for f in ctx["agent_findings"] if f["agent"] == "visual_analyst"),
        "No visual analysis available."
    )

    # Summarise CXformer features as a statistical description
    import numpy as np
    cxf = np.array(ctx.get("cxformer_features", [0]*384))
    top_dims  = int(np.sum(cxf > cxf.mean() + cxf.std()))
    feat_norm = float(np.linalg.norm(cxf))

    # Include KG findings and conflict information
    kg_info = ctx.get("kg_formatted", "No knowledge graph data available.")
    conflict_info = ""
    if ctx.get("conflict_detected"):
        conflict_info = f"\n⚠ CONFLICT DETECTED (severity: {ctx.get('conflict_severity', 'UNKNOWN')}):\n"
        conflict_info += "\n".join(f"  - {c}" for c in ctx.get("conflict_list", []))

    # Include modality weights if available
    modality_info = ""
    if ctx.get("modality_weights"):
        modality_info = "\nAdaptive fusion modality contributions:\n"
        for m, w in ctx["modality_weights"].items():
            modality_info += f"  {m}: {w}%\n"

    prompt = f"""You are a clinical AI reasoning assistant for a chest X-ray analysis system.

A chest X-ray has been analysed using four complementary sources:
1. ConvNeXt-Tiny: Local visual features (edges, textures, opacity patterns)
2. CXformer-S: Domain-specific features pretrained on 600,000+ chest X-rays
3. Radiology report text: Clinical findings from the associated report
4. Patient metadata: Age, symptoms, study information

Combined prediction: {ctx['prediction']}
Combined confidence: {round(ctx['confidence']*100, 1)}%
Probability of PNEUMONIA: {round(ctx.get('prob_pneumonia', ctx['confidence'])*100, 1)}%
Probability of NORMAL: {round(ctx.get('prob_normal', 1-ctx['confidence'])*100, 1)}%

Visual analyst finding: "{visual_finding}"

{kg_info}
{conflict_info}
{modality_info}

CXformer feature analysis:
- Feature vector norm: {feat_norm:.2f}
- Active feature dimensions (above mean+std): {top_dims} of 384
- Activation region: {ctx.get('activation_region', 'unknown')}

Task: In 3-4 sentences, provide:
1. Supporting evidence for the model's prediction
2. Any conflicting evidence or uncertainty factors
3. How the knowledge graph findings align or conflict with the prediction
4. What the adaptive attention weights reveal about which evidence is most reliable

Important: This is a RESEARCH prototype for paediatric chest X-rays. Do NOT replace clinical judgement.
Respond with only the 3-4 sentences, no headers."""

    finding = client.ask(prompt)
    ctx["agent_findings"].append({"agent": "clinical_reasoner", "finding": finding})
    print(f"[Agent 2 — Clinical Reasoner]\n{finding}\n")
    return ctx


def agent_evidence_aggregator_full(ctx: dict, client: GeminiClient) -> dict:
    """
    Synthesise all agent findings, KG verification, and conflict detection
    into a structured clinical report.
    """
    visual_finding   = next((f["finding"] for f in ctx["agent_findings"] if f["agent"] == "visual_analyst"), "")
    clinical_finding = next((f["finding"] for f in ctx["agent_findings"] if f["agent"] == "clinical_reasoner"), "")

    conflict_section = ""
    if ctx.get("conflict_detected"):
        conflict_section = f"""
⚠ CONFLICT ALERT (Severity: {ctx.get('conflict_severity', 'UNKNOWN')}):
{chr(10).join('- ' + c for c in ctx.get('conflict_list', []))}
"""

    modality_section = ""
    if ctx.get("modality_weights"):
        modality_section = "Adaptive fusion evidence weights:\n"
        for m, w in ctx["modality_weights"].items():
            modality_section += f"  {m}: {w}%\n"

    prompt = f"""You are a senior medical AI system aggregating findings from specialist sub-agents.

PATIENT: Paediatric chest X-ray (research dataset only)
PREDICTION: {ctx['prediction']} ({round(ctx['confidence']*100,1)}% confidence)
KG SCORE: {ctx.get('kg_score', 0.0):.3f} (range -1=normal to +1=pneumonia)
KG CONSISTENCY: {ctx.get('kg_consistency', 'UNKNOWN')}

AGENT FINDINGS:
Visual Analyst: "{visual_finding}"
Clinical Reasoner: "{clinical_finding}"
{conflict_section}
{modality_section}
Supporting KG findings: {', '.join(ctx.get('kg_supporting', [])) or 'none'}
Contradicting KG findings: {', '.join(ctx.get('kg_contradicting', [])) or 'none'}

Task: Produce a structured clinical summary with exactly these sections:

PRIMARY FINDING:
[One sentence describing the main observation]

SUPPORTING EVIDENCE:
[2-3 bullet points of specific evidence from the analysis]

KNOWLEDGE GRAPH VERIFICATION:
[One sentence on whether the KG findings support or contradict the prediction]

CONFLICT STATUS:
[State: NO CONFLICT or CONFLICT DETECTED — one sentence explaining]

CONFIDENCE ASSESSMENT:
[One sentence on reliability level and key uncertainties]

RECOMMENDATION:
[One of: "MONITOR - routine follow-up" | "REFER - clinical evaluation recommended" | "URGENT - immediate clinical review advised"]
[One sentence explaining the recommendation]

DISCLAIMER:
This is an AI research tool. Results must be verified by a qualified clinician before any clinical decision.

Respond with only the structured content above. Use the exact section headers."""

    report = client.ask(prompt)
    ctx["final_report"] = report
    ctx["agent_findings"].append({"agent": "evidence_aggregator", "finding": report})
    print(f"[Agent 3 — Evidence Aggregator]\n{report}\n")
    return ctx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_medreason(
    image_path: str,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-1.5-flash",
    save_output: bool = True,
) -> dict:
    """
    Full MedReason++ pipeline:
      1. Extract features + GradCAM
      2. Agent 1 — Visual Analyst
      3. Agent 2 — Clinical Reasoner
      4. Agent 3 — Evidence Aggregator

    Parameters
    ----------
    image_path     : path to chest X-ray JPEG
    gemini_api_key : Gemini API key (or set GEMINI_API_KEY env var)
    gemini_model   : Gemini model name
    save_output    : save JSON report to reports/agent_reports/

    Returns
    -------
    dict — SharedContext with all findings
    """
    from extract_features import FeatureExtractor

    print("=" * 60)
    print("  MedReason++ — Multi-Agent Clinical Reasoning")
    print("=" * 60)

    # Step 1: Extract features
    print("\n[Step 1/4] Extracting features and GradCAM...")
    extractor = FeatureExtractor()
    ctx = extractor.extract(image_path)
    print(f"  Prediction: {ctx['prediction']} ({ctx['confidence']*100:.1f}%)")
    print(f"  Activation: {ctx['activation_region']}")

    # Step 2-4: Agent chain
    print("\n[Step 2/4] Initialising Gemini agents...")
    client = GeminiClient(api_key=gemini_api_key, model=gemini_model)

    print("\n[Step 3/4] Running agent chain...")
    ctx = agent_visual_analyst(ctx, client)
    ctx = agent_clinical_reasoner(ctx, client)
    ctx = agent_evidence_aggregator(ctx, client)

    # Step 5: Save
    if save_output:
        out_dir = Path(__file__).resolve().parent.parent / "reports" / "agent_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(image_path).stem
        out_path = out_dir / f"{img_stem}_report.json"

        # Save without large base64 arrays
        save_ctx = {k: v for k, v in ctx.items()
                    if k not in ("convnext_features", "cxformer_features",
                                 "gradcam_heatmap_b64", "gradcam_overlay_b64")}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(save_ctx, f, indent=2)
        print(f"\n[Step 4/4] Report saved: {out_path}")

    print("\n" + "=" * 60)
    print("  MedReason++ analysis complete.")
    print("=" * 60)
    return ctx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MedReason++ multi-agent clinical reasoning."
    )
    parser.add_argument("--image",      required=True, help="Path to chest X-ray JPEG")
    parser.add_argument("--api_key",    default=None,  help="Gemini API key")
    parser.add_argument("--model",      default="gemini-1.5-flash")
    parser.add_argument("--no_save",    action="store_true")
    args = parser.parse_args()

    result = run_medreason(
        image_path=args.image,
        gemini_api_key=args.api_key,
        gemini_model=args.model,
        save_output=not args.no_save,
    )

    print("\n=== FINAL REPORT ===")
    print(result["final_report"])
