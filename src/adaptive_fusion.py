"""
src/adaptive_fusion.py
=======================
Adaptive Multimodal Fusion Model for MedReason++

Fuses FOUR input modalities:
    1. Image (ConvNeXt-Tiny)  → 768-dim visual features
    2. Image (CXformer-S)     → 384-dim chest-domain features
    3. Report text            → 384-dim semantic features
    4. Metadata               →  64-dim structured features

Architecture:
    Each branch is projected to a common dimension (256).
    An attention mechanism learns to weight each branch adaptively —
    so the model can rely more on image when text is unavailable,
    or trust text more when image features are ambiguous.

    proj_cx   (256) ─┐
    proj_cxf  (256) ─┤
    proj_text (256) ─┼──► Attention weights (softmax) ──► Weighted sum ──► 256-dim
    proj_meta  (64) ─┘                                                          │
                                                            Dropout(0.3) ──► Linear(256→2)

This design is called "Adaptive Late Fusion with Attention Gating."

⚠ DISCLAIMER: Research prototype. NOT for clinical use.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Adaptive Multimodal Fusion Model
# ---------------------------------------------------------------------------

class AdaptiveFusionModel(nn.Module):
    """
    Adaptive multimodal fusion of image + text + metadata features.

    Parameters
    ----------
    cx_dim    : int  ConvNeXt-Tiny feature dim (768)
    cxf_dim   : int  CXformer-S feature dim (384)
    text_dim  : int  Text encoder output dim (384)
    meta_dim  : int  Metadata encoder output dim (64)
    proj_dim  : int  Shared projection dim (256)
    num_classes: int Number of output classes (2)
    dropout   : float Dropout rate (0.3)
    """

    def __init__(
        self,
        cx_dim:     int = 768,
        cxf_dim:    int = 384,
        text_dim:   int = 384,
        meta_dim:   int = 64,
        proj_dim:   int = 256,
        num_classes: int = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.proj_dim = proj_dim

        # ── Projection layers for each modality ──────────────────────
        self.proj_cx = nn.Sequential(
            nn.Linear(cx_dim,   proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.proj_cxf = nn.Sequential(
            nn.Linear(cxf_dim,  proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.proj_meta = nn.Sequential(
            nn.Linear(meta_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())

        # ── Adaptive attention gate ───────────────────────────────────
        # Takes concatenated modality vectors and produces attention weights
        # over the 4 modalities — the model learns which to trust most
        self.attention_gate = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.Tanh(),
            nn.Linear(proj_dim, 4),   # 4 attention weights
            nn.Softmax(dim=-1),
        )

        # ── Final classifier ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_classes),
        )

    def forward(
        self,
        cx_feat:   torch.Tensor,          # [B, 768]
        cxf_feat:  torch.Tensor,          # [B, 384]
        text_feat: torch.Tensor,          # [B, 384]
        meta_feat: torch.Tensor,          # [B, meta_dim] (default [B, 64])
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the adaptive fusion model.

        Returns
        -------
        logits        : torch.Tensor  shape (B, num_classes)
        attn_weights  : torch.Tensor  shape (B, 4)  — interpretable modality weights
        """
        # Project each modality to shared dim
        p_cx   = self.proj_cx(cx_feat)    # [B, 256]
        p_cxf  = self.proj_cxf(cxf_feat)  # [B, 256]
        p_text = self.proj_text(text_feat) # [B, 256]
        p_meta = self.proj_meta(meta_feat) # [B, 256]

        # Compute adaptive attention weights over 4 modalities
        concat       = torch.cat([p_cx, p_cxf, p_text, p_meta], dim=-1)  # [B, 1024]
        attn_weights = self.attention_gate(concat)                         # [B, 4]

        # Weighted combination: each modality weighted by its attention score
        # attn_weights[:, 0] = weight for ConvNeXt
        # attn_weights[:, 1] = weight for CXformer
        # attn_weights[:, 2] = weight for text
        # attn_weights[:, 3] = weight for metadata
        fused = (
            attn_weights[:, 0:1] * p_cx   +
            attn_weights[:, 1:2] * p_cxf  +
            attn_weights[:, 2:3] * p_text +
            attn_weights[:, 3:4] * p_meta
        )  # [B, 256]

        logits = self.classifier(fused)   # [B, 2]
        return logits, attn_weights

    def get_modality_contributions(
        self, attn_weights: torch.Tensor
    ) -> dict[str, float]:
        """
        Human-readable modality contribution from attention weights.

        Parameters
        ----------
        attn_weights : torch.Tensor  shape (1, 4)

        Returns
        -------
        dict  {modality: contribution_percent}
        """
        w = attn_weights.squeeze(0).detach().cpu().tolist()
        return {
            "ConvNeXt-Tiny (local visual)":         round(w[0] * 100, 1),
            "CXformer-S (chest-domain)":             round(w[1] * 100, 1),
            "Radiology report text":                 round(w[2] * 100, 1),
            "Patient metadata":                      round(w[3] * 100, 1),
        }


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Detects disagreements between prediction sources:
      - Model prediction vs Knowledge Graph score
      - Image features vs Text features
      - High uncertainty despite high confidence

    Returns a structured conflict report.
    """

    def __init__(
        self,
        consistency_threshold: float = 0.3,
        confidence_suspicion_threshold: float = 0.95,
    ):
        self.consist_thresh  = consistency_threshold
        self.conf_thresh     = confidence_suspicion_threshold

    def detect(
        self,
        prediction:       str,
        confidence:       float,
        attn_weights:     dict[str, float],
        kg_score:         float,
        kg_consistency:   str,
        text_supports_pn: bool,
        image_supports_pn: bool,
    ) -> dict:
        """
        Run conflict detection across all evidence sources.

        Returns
        -------
        dict with keys:
            conflicts_found  : bool
            conflict_list    : list[str]
            severity         : str  (NONE / LOW / MEDIUM / HIGH)
            explanation      : str
        """
        conflicts = []

        # ── 1. Model vs Knowledge Graph ───────────────────────────────
        if kg_consistency == "INCONSISTENT":
            direction = "pneumonia" if kg_score > 0 else "normal"
            conflicts.append(
                f"Model predicts {prediction} but knowledge graph evidence "
                f"(score={kg_score:.2f}) points toward {direction}."
            )

        # ── 2. Image evidence vs Text evidence ────────────────────────
        if image_supports_pn != text_supports_pn:
            conflicts.append(
                f"Image features suggest {'pneumonia' if image_supports_pn else 'normal'} "
                f"but report text suggests {'pneumonia' if text_supports_pn else 'normal'}."
            )

        # ── 3. Very high confidence in borderline case ────────────────
        if confidence > self.conf_thresh and kg_consistency == "UNCERTAIN":
            conflicts.append(
                f"Model confidence ({confidence*100:.0f}%) is very high but "
                f"knowledge graph evidence is ambiguous (uncertain)."
            )

        # ── 4. Text modality being ignored ────────────────────────────
        text_weight = attn_weights.get("Radiology report text", 0.0)
        if text_weight < 10.0 and abs(kg_score) > 0.5:
            conflicts.append(
                f"Strong knowledge graph evidence (score={kg_score:.2f}) present in report "
                f"but model is giving low weight to text ({text_weight:.1f}%). "
                f"Consider reviewing the text evidence."
            )

        # ── Severity ──────────────────────────────────────────────────
        if len(conflicts) == 0:
            severity = "NONE"
        elif len(conflicts) == 1:
            severity = "LOW"
        elif len(conflicts) == 2:
            severity = "MEDIUM"
        else:
            severity = "HIGH"

        explanation = (
            "No conflicting evidence detected — prediction is well-supported."
            if not conflicts
            else "Conflicting evidence found. Clinical review is advised before acting on this result."
        )

        return {
            "conflicts_found": len(conflicts) > 0,
            "conflict_list":   conflicts,
            "severity":        severity,
            "explanation":     explanation,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = AdaptiveFusionModel()
    print(f"AdaptiveFusionModel parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    # Test forward pass with random features
    B = 2
    cx_feat   = torch.randn(B, 768)
    cxf_feat  = torch.randn(B, 384)
    text_feat = torch.randn(B, 384)
    meta_feat = torch.randn(B, 64)

    logits, attn = model(cx_feat, cxf_feat, text_feat, meta_feat)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Attention weights (sample): {attn[0].detach().tolist()}")
    print(f"Modality contributions: {model.get_modality_contributions(attn[0:1])}")

    # Test conflict detector
    cd = ConflictDetector()
    result = cd.detect(
        prediction="PNEUMONIA", confidence=0.92,
        attn_weights={"Radiology report text": 5.0, "ConvNeXt-Tiny (local visual)": 60.0,
                      "CXformer-S (chest-domain)": 30.0, "Patient metadata": 5.0},
        kg_score=-0.6, kg_consistency="INCONSISTENT",
        text_supports_pn=False, image_supports_pn=True,
    )
    print(f"\nConflict detection result:")
    print(f"  Conflicts found: {result['conflicts_found']}")
    print(f"  Severity: {result['severity']}")
    for c in result["conflict_list"]:
        print(f"  - {c}")
