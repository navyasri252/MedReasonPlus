import json
from pathlib import Path
base = Path(r'c:\Users\navya\Downloads\MedReasonPlus')

files = [
    ('ResNet-18 baseline (phase6)',       'processed/phase6/test_metrics.json'),
    ('ResNet-18 extended (phase6b)',      'processed/phase6b/test_metrics_default_thresh.json'),
    ('ConvNeXt-Tiny (local)',             'processed/convnext/test_metrics.json'),
    ('Hybrid 2-modal (hybrid folder)',    'processed/hybrid_convnext_cxformer/test_metrics.json'),
    ('CXformer-S',                        'processed/cxformer/test_metrics.json'),
]

for label, f in files:
    p = base / f
    if not p.exists():
        print(f'{label}: FILE NOT FOUND at {f}')
        print()
        continue
    d = json.load(p.open())
    cm = d.get('confusion_matrix', {})
    tn = cm.get('tn', 0); fp = cm.get('fp', 0)
    fn = cm.get('fn', 0); tp = cm.get('tp', 0)
    total = tn + fp + fn + tp
    acc_from_cm = round((tp + tn) / total, 4) if total > 0 else 0
    stored_acc  = d.get('accuracy', None)
    match = abs(acc_from_cm - stored_acc) < 0.001 if stored_acc is not None else False

    print(f'{label}')
    print(f'  accuracy={stored_acc}  precision={d.get("precision")}  recall={d.get("recall")}')
    print(f'  specificity={d.get("specificity")}  f1={d.get("f1")}  roc_auc={d.get("roc_auc")}')
    print(f'  CM: TN={tn} FP={fp} FN={fn} TP={tp}  (total={total})')
    print(f'  Acc from CM: {acc_from_cm}  Match: {match}')
    # verify specificity
    if (tn + fp) > 0:
        spec_check = round(tn / (tn + fp), 4)
        print(f'  Spec from CM: {spec_check}  Match: {abs(spec_check - d.get("specificity", 0)) < 0.001}')
    print()
