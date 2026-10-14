"""Inspect saved training history and replay frozen classifier; no sampling or fitting."""
from pathlib import Path
import json, hashlib
import torch, pandas as pd, numpy as np, joblib
ROOT=Path('/home/wyw/rstudio/MSI')
BASE=ROOT/'最终模型/data/stRNA/result/weighted_mlh1_transfer_seed2026'
OUT=ROOT/'最终模型/report/article/method_evidence_update_20261014/audit_spatial'
REPORT=ROOT/'最终模型/report/article/weighted_spatial_update'
MODEL=ROOT/'最终模型/model/reference/shared18_weighted_mlh1_2026/model.joblib'
sig=pd.read_csv(BASE/'reference/signature.csv',index_col=0)
r=torch.load(BASE/'reference/model/model.pt',map_location='cpu',weights_only=False)
ref=dict(checkpoint='reference/model/model.pt',top_keys=list(r),attr_keys=list(r['attr_dict']),history_keys=list(r['attr_dict']['history_']),is_trained=bool(r['attr_dict']['is_trained_']),n_genes=len(r['var_names']),reference_gene_order_matches_signature=bool(np.array_equal(r['var_names'],sig.index)),reference_history_recoverable_from_checkpoint=bool(r['attr_dict']['history_']),convergence_established=False)
rows=[]
for d in sorted(BASE.glob('VISDS*')):
 c=torch.load(d/'cell2location_model/model.pt',map_location='cpu',weights_only=False)
 h=c['attr_dict']['history_']['elbo_train']
 csv=pd.read_csv(d/'history_elbo_train.csv',index_col=0)
 assert np.array_equal(h.to_numpy(),csv.to_numpy())
 names=list(c['attr_dict']['factor_names_'])
 np.testing.assert_allclose(c['model_state_dict']['_model.cell_state'].cpu().numpy().T,sig.loc[c['var_names'],names].to_numpy(dtype=np.float32),atol=0,rtol=0)
 rows.append(dict(sample=d.name,history_keys=list(c['attr_dict']['history_']),history_epochs=len(h),history_csv_matches_checkpoint=True,signature_matches_checkpoint=True,is_trained=bool(c['attr_dict']['is_trained_'])))
 print(d.name,'checkpoint PASS',flush=True)
a=joblib.load(MODEL);f=pd.read_csv(REPORT/'spatial_classifier_features.csv',index_col=0)
pred=pd.read_csv(REPORT/'section_predictions.csv').set_index('sampleID')
score=a['pipeline'].predict_proba(f[a['features']])[:,1]
assert set(f.index)==set(pred.index)
np.testing.assert_allclose(score,pred.loc[f.index,'score'],atol=1e-12,rtol=1e-12)
classifier=dict(features=list(a['features']),pipeline_steps={n:type(step).__name__ for n,step in a['pipeline'].steps},model_solver=a['pipeline']['model'].solver,model_shrinkage=a['pipeline']['model'].shrinkage,replayed_prediction_max_abs_error=float(np.max(np.abs(score-pred.loc[f.index,'score']))),source_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest())
(OUT/'checkpoint_history_and_classifier_audit.json').write_text(json.dumps(dict(reference=ref,sections=rows,frozen_classifier=classifier),ensure_ascii=False,indent=2))
print(json.dumps(dict(reference=ref,frozen_classifier=classifier),ensure_ascii=False,indent=2),flush=True)
