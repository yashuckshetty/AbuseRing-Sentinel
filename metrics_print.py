import json
m = json.load(open('evals/metrics.json'))
for r in m:
    mdl = r.get('model','?')
    spl = r.get('split','?')
    print(mdl, spl, 'P-ac=', r.get('precision_abusive','N/A'), 'R-ac=', r.get('recall_abusive','N/A'), 'F1=', r.get('f1_abusive','N/A'), 'AUC=', r.get('roc_auc_macro','N/A'))