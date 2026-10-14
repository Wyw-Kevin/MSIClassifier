"""Grouped OOF MLH1-loss classification and external MSS-only false flags."""
from pathlib import Path
import json
import joblib
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '最终模型/data/scRNA/result/diagnostics'
OUT = BASE / 'mmr4_loss_vs_mlh1_001_unique2026'
GENES = ['MLH1','MSH2','MSH6','PMS2']
EXT = BASE / 'infercnvpy_stromal_mlh1_impact_q95_external_unique_seed2026'

def build(ds):
    path = ROOT / (ds+'.h5ad') if ds!='GSE200997' else ROOT/'最终模型/data/scRNA/external/GSE200997/GSE200997.h5ad'
    a = ad.read_h5ad(path, backed='r')
    assert a.obs_names.is_unique
    obs = a.obs.copy()
    if ds=='GSE178341':
        ids = pd.read_csv(BASE/'infercnvpy_stromal_mlh1_impact_q95_training_unique_seed2026/GSE178341_selected_epithelial_cells.csv.gz',index_col=0).index
        sample_col = 'sample'
        audit = BASE/'infercnvpy_stromal_mlh1_impact_q95_training_unique_seed2026/GSE178341_sample_MLH1_stromal_infercnvpy.csv'
    else:
        ids = pd.read_csv(EXT/f'{ds}_selected_epithelial_cells.csv.gz',index_col=0).index
        sample_col = 'Sample'
        audit = EXT/f'{ds}_sample_MLH1_stromal_infercnvpy.csv'
    assert ids.is_unique and ids.isin(obs.index).all()
    loc = [np.flatnonzero(a.var_names==g) for g in GENES]
    assert all(len(v)==1 for v in loc)
    # Read one gene at a time to support backed CSR and duplicated other genes.
    expression = {}
    for g,j in zip(GENES,loc):
        v = a[:,int(j[0])].X
        expression[g] = np.asarray(v.toarray() if sparse.issparse(v) else v).ravel()
    cells = pd.DataFrame(expression,index=obs.index).loc[ids]
    cells['sampleID'] = obs.loc[ids,sample_col].astype(str)
    means = cells.groupby('sampleID')[GENES].mean()
    original = pd.read_csv(audit).set_index('sampleID')
    table = original[['MMR','n_malignant_epithelial']].join(means)
    np.testing.assert_allclose(table.MLH1,original.mean_MLH1_epithelial_new,atol=1e-7,equal_nan=True)
    if ds=='GSE178341':
        labels = obs.groupby('sample',observed=True).MLH1.first()
        table['stored_label'] = table.index.map(labels)
        table['loss'] = (table.MMR.eq('MSI')&table.stored_label.eq('loss')).astype(int)
        ledger = pd.read_csv(BASE/'shared18_same_sample_benchmark_seed2026/sample_ledger.csv')
        ledger = ledger[ledger.dataset.eq(ds)].set_index('sampleID')
        table = table.join(ledger[['patientID','fold']])
    a.file.close()
    table.to_csv(OUT/f'{ds}_four_gene_features.csv')
    return table

def model():
    return make_pipeline(StandardScaler(),LogisticRegression(C=1,solver='liblinear',random_state=2026,max_iter=2000))

def main():
    OUT.mkdir(exist_ok=True)
    train = build('GSE178341')
    assert len(train)==64 and train.loss.sum()==22 and train[GENES].notna().all().all()
    predictions = train[['MMR','stored_label','loss','patientID','fold']].copy()
    predictions['MLH1_score'] = -train.MLH1
    predictions['MLH1_prediction'] = train.MLH1.lt(.01).astype(int)
    coefficients = []
    for fold in sorted(train.fold.unique()):
        fit = train[train.fold.ne(fold)]; test = train[train.fold.eq(fold)]
        assert not set(fit.patientID)&set(test.patientID)
        m = model().fit(fit[GENES],fit.loss)
        predictions.loc[test.index,'MMR4_score'] = m.predict_proba(test[GENES])[:,1]
        coefficients.append(dict(fold=int(fold),intercept=float(m[-1].intercept_[0]),**dict(zip(GENES,m[-1].coef_[0]))))
    predictions['MMR4_prediction'] = predictions.MMR4_score.ge(.5).astype(int)
    predictions.to_csv(OUT/'training_grouped_OOF_predictions.csv')
    metrics=[]
    for scope,part in [('all64',predictions),('MSI_only',predictions[predictions.MMR.eq('MSI')])]:
        for name in ['MLH1','MMR4']:
            y=part.loss; pred=part[name+'_prediction']
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            metrics.append(dict(scope=scope,method=name,n=len(part),AUROC=roc_auc_score(y,part[name+'_score']),
                 accuracy=accuracy_score(y,pred),BA=balanced_accuracy_score(y,pred),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
    pd.DataFrame(metrics).to_csv(OUT/'training_metrics.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'fold_coefficients.csv',index=False)
    final = model().fit(train[GENES],train.loss)
    joblib.dump(dict(pipeline=final,features=GENES,target='stored_MLH1_loss',threshold=.5,seed=2026),OUT/'MMR4_loss_model.joblib')
    records=[]
    for ds in ['GSE132465','GSE200997']:
        x=build(ds); x=x[x.MMR.eq('MSS')].copy()
        assert x[GENES].notna().all().all()
        x['MMR4_loss_probability']=final.predict_proba(x[GENES])[:,1]
        x['MMR4_loss_flag']=x.MMR4_loss_probability.ge(.5)
        x['MLH1_001_loss_flag']=x.MLH1.lt(.01)
        x['dataset']=ds
        records.append(x.reset_index())
    external=pd.concat(records,ignore_index=True)
    external.to_csv(OUT/'external_MSS_only_predictions.csv',index=False)
    summary=external.groupby('dataset').agg(MSS_samples=('sampleID','size'),MMR4_loss_flags=('MMR4_loss_flag','sum'),MLH1_001_loss_flags=('MLH1_001_loss_flag','sum'))
    summary.to_csv(OUT/'external_MSS_false_flags.csv')
    (OUT/'manifest.json').write_text(json.dumps(dict(seed=2026,training_target='h5ad stored MLH1 loss versus express; MSS assigned express',
       model='Z-score + L2 logistic, C=1, unweighted, default threshold 0.5',
       evaluation='existing patient-grouped five folds; no parameter or threshold search',
       training_provenance='unique-cell corrected stromal-only q95; PCA/neighbor/Leiden seeds 2026',
       external_provenance='unique-cell corrected stromal-only q95, candidates unchanged',
       limitation='loss predictions are not independently validated MSI calls; labels require source verification; cohort-level CNV is not fold-isolated'),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False)); print(summary.to_string())
    print(external[external.MMR4_loss_flag|external.MLH1_001_loss_flag][['dataset','sampleID','MLH1','MMR4_loss_probability','MMR4_loss_flag','MLH1_001_loss_flag']].to_string(index=False))

if __name__=='__main__':
    main()
