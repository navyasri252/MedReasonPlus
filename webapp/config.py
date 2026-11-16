# webapp/config.py
# ─────────────────────────────────────────────────────────────
# Project configuration — keep this file out of version control
# Never expose GEMINI_API_KEY in the frontend HTML/JS
# ─────────────────────────────────────────────────────────────
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "auto"
