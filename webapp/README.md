# MedReason++ Showcase Website

## Quick Start

The MedReason++ website uses an evidence-first interface for the chest X-ray research prototype. It brings image analysis, optional report text, patient context, and visual explanations into one workspace. Model performance benchmark figures are intentionally not presented as clinical claims on the website.

```bash
# 1. Install dependencies (from project root)
pip install -r requirements.txt

# 2. Start the web server
cd webapp
uvicorn app:app --reload --port 8000

# 3. Open in browser
# http://localhost:8000
```

## Using the Demo

### Without Gemini API key
Upload a chest X-ray or click **Load Sample Images**.
You get:
- NORMAL / PNEUMONIA prediction with confidence
- GradCAM++ heatmap showing which lung region activated
- Probability bars for both classes

### With Gemini API key
Enter your key in the API Key field, then analyse.
You get everything above **plus**:
- Agent 1 — Visual Analyst description of the heatmap
- Agent 2 — Clinical Reasoner evidence assessment
- Agent 3 — Structured recommendation (MONITOR / REFER / URGENT)

## Getting a Gemini API Key
1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API key**
3. Copy and paste into the web form

## File Structure
```
webapp/
├── app.py              ← FastAPI backend
├── templates/
│   └── index.html      ← Main page
└── static/
    ├── css/style.css
    └── js/app.js
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Main webpage |
| GET | `/health` | Health check |
| POST | `/analyse` | Analyse uploaded image |
| POST | `/analyse-sample` | Analyse a built-in sample |
| GET | `/sample-images` | List available sample images |
| GET | `/gradcam/{file}` | Serve GradCAM PNG files |

## ⚠ Disclaimer
Research prototype only. NOT validated for clinical use.
Paediatric chest X-rays (Kermany et al. dataset).
