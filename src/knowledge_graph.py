"""
src/knowledge_graph.py
=======================
Lightweight Medical Knowledge Graph for MedReason++

This is a structured rule-based knowledge graph encoded as Python data
structures (no external graph database required).

Structure:
    Finding ──► Evidence weight ──► Diagnosis
    Finding ──► Contradicts     ──► Diagnosis

Sources:
    Clinical findings adapted from standard chest X-ray interpretation
    guidelines (not patient-specific clinical advice).

Usage:
    kg = MedicalKnowledgeGraph()
    result = kg.verify(report_text, prediction, confidence)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Knowledge graph nodes
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A clinical finding that can be observed in a chest X-ray report."""
    name: str
    keywords: list[str]           # regex patterns to detect in report text
    supports_pneumonia: float     # 0.0 → 1.0 (strength of evidence for pneumonia)
    contradicts_pneumonia: float  # 0.0 → 1.0 (strength of evidence against)
    description: str              # clinical explanation


@dataclass
class KGVerificationResult:
    """Result of verifying a prediction against the knowledge graph."""
    prediction: str               # PNEUMONIA or NORMAL
    confidence: float
    supporting_findings: list[str]
    contradicting_findings: list[str]
    neutral_findings: list[str]
    kg_score: float               # -1 (strong normal) to +1 (strong pneumonia)
    consistency: str              # CONSISTENT / INCONSISTENT / UNCERTAIN
    conflict_detected: bool
    conflict_reason: Optional[str]
    recommendation_modifier: str  # STRENGTHEN / WEAKEN / NEUTRAL


# ---------------------------------------------------------------------------
# Knowledge graph definition
# ---------------------------------------------------------------------------

PNEUMONIA_KNOWLEDGE_GRAPH: list[Finding] = [

    # ── Strong positive findings (support pneumonia) ───────────────────
    Finding(
        name="Lobar consolidation",
        keywords=[r"lobar\s+consolidat", r"airspace\s+disease\s+consistent\s+with\s+pneumonia",
                  r"there\s+is\s+(?:a\s+)?(?:right|left|bilateral)\s+(?:lower|upper|middle)?\s*(?:lobe\s+)?pneumonia"],
        supports_pneumonia=0.95,
        contradicts_pneumonia=0.0,
        description="Lobar consolidation is the hallmark radiographic finding of bacterial pneumonia."
    ),
    Finding(
        name="Focal airspace opacity",
        keywords=[r"focal\s+(?:airspace\s+)?opacity", r"focal\s+consolidat",
                  r"airspace\s+disease\s+(?:identified|noted|seen|present)"],
        supports_pneumonia=0.80,
        contradicts_pneumonia=0.0,
        description="Focal airspace opacity suggests localised infection or inflammation."
    ),
    Finding(
        name="Infiltrate",
        keywords=[r"infiltrat", r"perihilar\s+opacity"],
        supports_pneumonia=0.70,
        contradicts_pneumonia=0.0,
        description="Pulmonary infiltrates are commonly associated with pneumonia."
    ),
    Finding(
        name="Increased interstitial markings",
        keywords=[r"increased\s+interstitial", r"interstitial\s+opacit", r"interstitial\s+changes"],
        supports_pneumonia=0.50,
        contradicts_pneumonia=0.0,
        description="Interstitial markings may suggest viral or atypical pneumonia."
    ),

    # ── Strong negative findings (contradict pneumonia) ────────────────
    Finding(
        name="Clear lungs",
        keywords=[r"lungs\s+(?:are\s+)?(?:grossly\s+)?clear",
                  r"clear\s+without\s+focal\s+(?:airspace|opacit|consolidat)",
                  r"no\s+focal\s+(?:airspace\s+)?consolidat",
                  r"free\s+of\s+focal\s+airspace\s+disease"],
        supports_pneumonia=0.0,
        contradicts_pneumonia=0.95,
        description="Clear lung fields strongly argue against pneumonia."
    ),
    Finding(
        name="No pneumonia explicit",
        keywords=[r"no\s+pneumonia", r"no\s+active\s+pneumonia",
                  r"no\s+focal\s+air\s*space\s+opacit\w+\s+to\s+suggest"],
        supports_pneumonia=0.0,
        contradicts_pneumonia=0.98,
        description="Explicit radiologist statement of no pneumonia."
    ),
    Finding(
        name="No focal consolidation",
        keywords=[r"no\s+focal\s+consolidat", r"without\s+focal\s+airspace\s+disease",
                  r"no\s+evidence\s+of\s+(?:focal\s+)?consolidat"],
        supports_pneumonia=0.0,
        contradicts_pneumonia=0.85,
        description="Absence of consolidation makes pneumonia unlikely."
    ),
    Finding(
        name="Normal cardiomediastinal silhouette",
        keywords=[r"(?:normal|unremarkable)\s+(?:cardio\s*mediastinal|mediastinal)\s+(?:silhouette|contour)",
                  r"within\s+normal\s+limits.*mediastin"],
        supports_pneumonia=0.0,
        contradicts_pneumonia=0.30,
        description="Normal mediastinum is a weak negative indicator — most pneumonias don't involve the mediastinum."
    ),

    # ── Uncertain / hedged findings ────────────────────────────────────
    Finding(
        name="Atelectasis",
        keywords=[r"atelectasis", r"subsegmental\s+atelectasis", r"platelike\s+atelectasis"],
        supports_pneumonia=0.30,
        contradicts_pneumonia=0.10,
        description="Atelectasis can mimic or co-exist with pneumonia; not specific alone."
    ),
    Finding(
        name="Pleural effusion",
        keywords=[r"pleural\s+effusion", r"pleural\s+fluid"],
        supports_pneumonia=0.40,
        contradicts_pneumonia=0.0,
        description="Pleural effusion can accompany complicated pneumonia."
    ),
    Finding(
        name="No pleural effusion",
        keywords=[r"no\s+pleural\s+effusion", r"no\s+pleural\s+fluid",
                  r"pleural\s+spaces\s+(?:are\s+)?clear"],
        supports_pneumonia=0.0,
        contradicts_pneumonia=0.20,
        description="Absence of effusion slightly argues against complicated pneumonia."
    ),
    Finding(
        name="Hyperinflation",
        keywords=[r"hyperinflat", r"hyperexpand", r"flattening\s+of\s+the\s+diaphragm"],
        supports_pneumonia=0.15,
        contradicts_pneumonia=0.10,
        description="Hyperinflation suggests obstructive disease (COPD/asthma), but pneumonia can occur on top."
    ),
    Finding(
        name="Granulomatous disease",
        keywords=[r"granuloma", r"granulomatous", r"calcified.*nodule"],
        supports_pneumonia=0.10,
        contradicts_pneumonia=0.0,
        description="Old granulomatous changes indicate prior infection, not active pneumonia."
    ),
]


# ---------------------------------------------------------------------------
# Medical Knowledge Graph class
# ---------------------------------------------------------------------------

class MedicalKnowledgeGraph:
    """
    Rule-based medical knowledge graph for chest X-ray pneumonia verification.

    Verifies whether a model's prediction is consistent with the clinical
    findings mentioned in the associated radiology report.
    """

    def __init__(self):
        self.findings = PNEUMONIA_KNOWLEDGE_GRAPH

    def extract_findings(self, report_text: str) -> list[Finding]:
        """Find all knowledge graph nodes whose keywords match the report."""
        matched = []
        text = report_text.lower()
        for finding in self.findings:
            for kw in finding.keywords:
                if re.search(kw, text, re.IGNORECASE):
                    matched.append(finding)
                    break
        return matched

    def compute_kg_score(self, matched_findings: list[Finding]) -> float:
        """
        Compute a composite pneumonia score from matched findings.
        Score range: -1.0 (strong normal) to +1.0 (strong pneumonia)
        """
        if not matched_findings:
            return 0.0  # no evidence either way

        pos_score = sum(f.supports_pneumonia for f in matched_findings)
        neg_score = sum(f.contradicts_pneumonia for f in matched_findings)

        # Normalise to [-1, +1]
        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return round((pos_score - neg_score) / total, 4)

    def verify(
        self,
        report_text: str,
        prediction: str,
        confidence: float,
    ) -> KGVerificationResult:
        """
        Verify a model prediction against the knowledge graph.

        Parameters
        ----------
        report_text : str   radiology report text
        prediction  : str   "PNEUMONIA" or "NORMAL"
        confidence  : float 0.0–1.0

        Returns
        -------
        KGVerificationResult
        """
        matched = self.extract_findings(report_text)

        supporting     = [f.name for f in matched if f.supports_pneumonia > 0.3]
        contradicting  = [f.name for f in matched if f.contradicts_pneumonia > 0.3]
        neutral        = [f.name for f in matched
                         if f.supports_pneumonia <= 0.3 and f.contradicts_pneumonia <= 0.3]

        kg_score = self.compute_kg_score(matched)

        # Determine consistency
        pred_is_pneumonia = prediction == "PNEUMONIA"

        if pred_is_pneumonia and kg_score >= 0.3:
            consistency = "CONSISTENT"
            modifier    = "STRENGTHEN"
        elif not pred_is_pneumonia and kg_score <= -0.3:
            consistency = "CONSISTENT"
            modifier    = "STRENGTHEN"
        elif pred_is_pneumonia and kg_score <= -0.5:
            consistency = "INCONSISTENT"
            modifier    = "WEAKEN"
        elif not pred_is_pneumonia and kg_score >= 0.5:
            consistency = "INCONSISTENT"
            modifier    = "WEAKEN"
        else:
            consistency = "UNCERTAIN"
            modifier    = "NEUTRAL"

        # Conflict detection
        conflict_detected = False
        conflict_reason   = None

        if supporting and contradicting:
            conflict_detected = True
            conflict_reason = (
                f"Report contains both pneumonia-supporting findings "
                f"({', '.join(supporting[:2])}) and "
                f"contradicting findings ({', '.join(contradicting[:2])})."
            )
        elif consistency == "INCONSISTENT":
            conflict_detected = True
            kg_dir = "pneumonia" if kg_score > 0 else "normal"
            conflict_reason = (
                f"Model predicts {prediction} but knowledge graph score ({kg_score:.2f}) "
                f"indicates {kg_dir}. "
                f"Contradicting findings: {', '.join(contradicting) or 'none'}. "
                f"Supporting findings: {', '.join(supporting) or 'none'}."
            )

        return KGVerificationResult(
            prediction=prediction,
            confidence=confidence,
            supporting_findings=supporting,
            contradicting_findings=contradicting,
            neutral_findings=neutral,
            kg_score=kg_score,
            consistency=consistency,
            conflict_detected=conflict_detected,
            conflict_reason=conflict_reason,
            recommendation_modifier=modifier,
        )

    def format_for_agent(self, result: KGVerificationResult) -> str:
        """Format the KG verification result as text for the Gemini agent."""
        lines = [
            f"Knowledge Graph Verification:",
            f"  KG score: {result.kg_score:.3f} (range -1=normal to +1=pneumonia)",
            f"  Consistency with prediction: {result.consistency}",
        ]
        if result.supporting_findings:
            lines.append(f"  Supporting findings: {', '.join(result.supporting_findings)}")
        if result.contradicting_findings:
            lines.append(f"  Contradicting findings: {', '.join(result.contradicting_findings)}")
        if result.conflict_detected:
            lines.append(f"  ⚠ CONFLICT DETECTED: {result.conflict_reason}")
        lines.append(f"  Recommendation modifier: {result.recommendation_modifier}")
        return "\n".join(lines)
