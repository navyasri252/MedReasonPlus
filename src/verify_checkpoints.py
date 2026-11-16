import torch, sys, torch.nn as nn
sys.path.insert(0, r'c:\Users\navya\Downloads\MedReasonPlus\src')
from torchvision.models import convnext_tiny
from transformers import AutoModel
from pathlib import Path

PROJ = 256

# ── 1. Verify CXformer checkpoint ──────────────────────────────────
print('=== Verifying CXformer-S checkpoint ===')
cxf_bb = AutoModel.from_pretrained('m42-health/CXformer-small')

class CXformerClassifier(nn.Module):
    def __init__(self, bb, hdim=384):
        super().__init__()
        self.bb   = bb
        self.head = nn.Linear(hdim, 2)
    def forward(self, x):
        return self.head(self.bb(pixel_values=x).last_hidden_state[:, 0, :])

cxf_model = CXformerClassifier(cxf_bb)
cxf_ckpt  = Path(r'c:\Users\navya\Downloads\MedReasonPlus\processed\cxformer\best_model.pt')
state     = torch.load(cxf_ckpt, map_location='cpu', weights_only=True)
print('  Keys in CXformer state dict:', len(state))
print('  Sample keys:', list(state.keys())[:6])
try:
    missing, unexpected = cxf_model.load_state_dict(state, strict=False)
    print('  CXformer checkpoint: LOADED (strict=False)')
    print('  Missing keys:', len(missing), '  Unexpected keys:', len(unexpected))
    # Quick forward pass
    x = torch.zeros(1, 3, 518, 518)
    with torch.no_grad():
        out = cxf_model(x)
    print('  Forward pass shape:', out.shape, '(should be [1, 2])')
except Exception as e:
    print('  ERROR loading CXformer:', type(e).__name__, str(e)[:200])

# ── 2. Verify Hybrid checkpoint ────────────────────────────────────
print()
print('=== Verifying Hybrid checkpoint ===')
hyb_ckpt  = Path(r'c:\Users\navya\Downloads\MedReasonPlus\processed\hybrid_convnext_cxformer\best_hybrid_model.pth')
hyb_state = torch.load(hyb_ckpt, map_location='cpu', weights_only=True)
print('  Keys in Hybrid state dict:', len(hyb_state))
prefixes = sorted({k.split('.')[0] for k in hyb_state.keys()})
print('  Top-level prefixes:', prefixes)

# Rebuild hybrid model to test load
_cx_base = convnext_tiny(weights=None)
class ConvNeXtExtractor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.avgpool  = base.avgpool
    def forward(self, x):
        return self.avgpool(self.features(x)).flatten(1)

cx_enc = ConvNeXtExtractor(_cx_base)
for p in cx_enc.parameters(): p.requires_grad = False

_cxf_bb2 = AutoModel.from_pretrained('m42-health/CXformer-small')
class CXformerExtractor(nn.Module):
    def __init__(self, bb): super().__init__(); self.bb = bb
    def forward(self, x): return self.bb(pixel_values=x).last_hidden_state[:, 0, :]
cxf_enc = CXformerExtractor(_cxf_bb2)
for p in cxf_enc.parameters(): p.requires_grad = False

class HybridFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cx_enc   = cx_enc
        self.cxf_enc  = cxf_enc
        self.proj_cx  = nn.Sequential(nn.Linear(768, PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.proj_cxf = nn.Sequential(nn.Linear(384, PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(PROJ * 2, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, cx_img, cxf_img):
        with torch.no_grad():
            f_cx  = self.cx_enc(cx_img)
            f_cxf = self.cxf_enc(cxf_img)
        return self.classifier(torch.cat([self.proj_cx(f_cx), self.proj_cxf(f_cxf)], dim=1))

hybrid = HybridFusionModel()
try:
    missing, unexpected = hybrid.load_state_dict(hyb_state, strict=False)
    print('  Hybrid checkpoint: LOADED (strict=False)')
    print('  Missing keys (from HF version diff):', len(missing))
    print('  Unexpected keys:', len(unexpected))
    # Quick forward pass
    cx_x  = torch.zeros(1, 3, 224, 224)
    cxf_x = torch.zeros(1, 3, 518, 518)
    with torch.no_grad():
        out = hybrid(cx_x, cxf_x)
    print('  Forward pass shape:', out.shape, '(should be [1, 2])')
except Exception as e:
    print('  ERROR loading Hybrid:', type(e).__name__, str(e)[:300])
