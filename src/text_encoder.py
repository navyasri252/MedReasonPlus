"""
src/text_encoder.py
====================
Radiology Report Text Encoder for MedReason++

Encodes free-text radiology reports into fixed-size dense vectors
using sentence-transformers (all-MiniLM-L6-v2, 384-dim).

This model was chosen because:
  - Lightweight (80 MB) — runs on CPU without issue
  - Trained on 1B+ sentence pairs — strong general semantic embedding
  - 384-dim output — directly compatible with CXformer-S feature dim

The text features are used in the adaptive multimodal fusion model
alongside image features from ConvNeXt-Tiny and CXformer-S.

Usage:
    encoder = ReportTextEncoder()
    vec = encoder.encode("The lungs are clear. No focal consolidation.")
    # vec.shape == (384,)
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Report text cleaner
# ---------------------------------------------------------------------------

def clean_report(text: str) -> str:
    """
    Normalise IU X-ray report text before encoding:
      - Replace XXXX de-identification tokens with [REDACTED]
      - Collapse multiple spaces / newlines
      - Strip leading/trailing whitespace
    """
    text = re.sub(r"XXXX", "[REDACTED]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ReportTextEncoder
# ---------------------------------------------------------------------------

class ReportTextEncoder:
    """
    Encodes radiology report text into dense feature vectors.

    Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) loaded once
    and reused for all subsequent encodings.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model name (default: all-MiniLM-L6-v2)
    device : str
        'cpu' or 'cuda'
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    OUTPUT_DIM = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load(model_name)

    def _load(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[TextEncoder] Loading {model_name} on {self.device}…")
            m = SentenceTransformer(model_name, device=self.device)
            print(f"[TextEncoder] Loaded OK — output dim: {self.OUTPUT_DIM}")
            return m
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            ) from exc

    def encode(self, report_text: str) -> np.ndarray:
        """
        Encode a single report into a 384-dim float32 numpy array.

        Parameters
        ----------
        report_text : str

        Returns
        -------
        np.ndarray  shape (384,)
        """
        cleaned = clean_report(report_text)
        vec = self._model.encode(cleaned, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, reports: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of reports in batches.

        Parameters
        ----------
        reports    : list[str]
        batch_size : int

        Returns
        -------
        np.ndarray  shape (N, 384)
        """
        cleaned = [clean_report(r) for r in reports]
        vecs = self._model.encode(
            cleaned,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(cleaned) > 100,
        )
        return vecs.astype(np.float32)

    def encode_tensor(self, report_text: str) -> torch.Tensor:
        """
        Encode and return as a PyTorch tensor with shape (1, 384).
        """
        vec = self.encode(report_text)
        return torch.from_numpy(vec).unsqueeze(0)


# ---------------------------------------------------------------------------
# TextProjectionHead — plugs text features into the fusion model
# ---------------------------------------------------------------------------

class TextProjectionHead(nn.Module):
    """
    Projects 384-dim text features to PROJ_DIM for fusion.

    Architecture:
        Linear(384 → proj_dim) → LayerNorm → GELU → Dropout(0.1)
    """

    def __init__(self, input_dim: int = 384, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Metadata encoder
# ---------------------------------------------------------------------------

class MetadataEncoder(nn.Module):
    """
    Encodes structured patient/study metadata into a dense vector.

    Expected metadata fields (all optional — missing values are 0.0):
        age_years      : float  (normalised to 0-1 by /100)
        has_symptoms   : bool   (1.0 if patient has respiratory symptoms)
        is_followup    : bool   (1.0 if this is a follow-up study)
        study_image_count : int (number of views, normalised by /5)
        report_word_count : int (normalised by /150)

    Architecture:
        Linear(5 → 32) → ReLU → Linear(32 → proj_dim)
    """

    INPUT_DIM  = 5
    HIDDEN_DIM = 32

    def __init__(self, proj_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.HIDDEN_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

    def forward(self, metadata_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        metadata_vec : torch.Tensor  shape (B, 5)

        Returns
        -------
        torch.Tensor  shape (B, proj_dim)
        """
        return self.encoder(metadata_vec)

    @staticmethod
    def build_vector(
        age_years: Optional[float] = None,
        has_symptoms: bool = False,
        is_followup: bool = False,
        study_image_count: int = 2,
        report_word_count: int = 30,
    ) -> torch.Tensor:
        """
        Build a normalised metadata tensor from raw values.

        Returns
        -------
        torch.Tensor  shape (1, 5)
        """
        age_norm   = min((age_years or 10.0) / 100.0, 1.0)  # default 10 (paediatric)
        symp_flag  = 1.0 if has_symptoms else 0.0
        follow_flag= 1.0 if is_followup  else 0.0
        img_norm   = min(study_image_count / 5.0, 1.0)
        wc_norm    = min(report_word_count / 150.0, 1.0)

        vec = torch.tensor(
            [[age_norm, symp_flag, follow_flag, img_norm, wc_norm]],
            dtype=torch.float32
        )
        return vec


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    encoder = ReportTextEncoder()

    reports = [
        "The lungs are clear. No focal consolidation. No pleural effusion.",
        "There is a right lower lobe pneumonia. No pleural effusion.",
        "Left lower lobe airspace disease identified. Consistent with pneumonia.",
    ]

    for r in reports:
        vec = encoder.encode(r)
        print(f"Report: {r[:60]}…")
        print(f"  Vector shape: {vec.shape}  norm: {np.linalg.norm(vec):.4f}")
        print()

    # Test metadata encoder
    meta_enc = MetadataEncoder(proj_dim=64)
    meta_vec = MetadataEncoder.build_vector(age_years=8, has_symptoms=True)
    out = meta_enc(meta_vec)
    print(f"Metadata encoder output shape: {out.shape}")

    # Test text projection head
    proj = TextProjectionHead(input_dim=384, proj_dim=256)
    t    = torch.from_numpy(encoder.encode_batch(reports))
    out2 = proj(t)
    print(f"Text projection output shape: {out2.shape}")
