"""Audit and visualize completed spatial reconstruction and frozen classifier transfer."""
from pathlib import Path
import json
import re
import hashlib
import subprocess
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FINAL=Path(__file__).resolve().parents[1]
SOURCE=FINAL/'data/stRNA/result/weighted_mlh1_transfer_seed2026'
OUT=FINAL/'report/article/weighted_spatial_update'
MODEL=FINAL/'model/reference/shared18_weighted_mlh1_2026/model.joblib'
TRAIN=FINAL/'data/scRNA/result/diagnostics/shared18_malignant_fraction_weighted_mlh1_seed2026'
VARIANT='malignant_fraction_weighted_MLH1_refit'

def main():
    OUT.mkdir(exist_ok=True)
    directories=sorted(p.parent for p in SOURCE.glob('VISDS*/complete.json'))
    assert len(directories)==14, 'Wait until all 14 reconstructions complete'
    artifact=joblib.load(MODEL); features=artifact['features']; classifier=artifact['pipeline']
    plan=json.loads((SOURCE/'analysis_plan.json').read_text())
    model_hash=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert model_hash==plan['classifier_sha256'], 'Frozen classifier changed during reconstruction'
    assert len(features)==19 and classifier.classes_[1]==1
    train=pd.read_csv(TRAIN/f'GSE178341_{VARIANT}_features.csv',index_col=0)
    source_labels=pd.read_csv(TRAIN/'sample_counts_and_weights.csv').query("dataset == 'GSE178341'").set_index('sampleID')
    ecology=clone(classifier).fit(train[features[:-1]],source_labels.loc[train.index].MMR.eq('MSI'))
    joblib.dump(dict(pipeline=ecology,features=features[:-1],training_dataset='GSE178341'),OUT/'ecology_only_comparator.joblib')
    records=[];spotdata={};frames=[];convergence=[]
    for directory in directories:
        metadata=json.loads((directory/'complete.json').read_text())
        assert metadata['official_expression_formula_verified'] and metadata['seed']==2026
        assert metadata['spatial_epochs']==500 and (directory/'history_elbo_train.csv').exists()
        data=pd.read_csv(directory/'classifier_features.csv',index_col=0)
        assert len(data)==1 and np.isfinite(data[features].to_numpy()).all()
        spot=pd.read_csv(directory/'spot_mlh1_and_abundance.csv.gz',index_col=0)
        abundance=pd.read_csv(directory/'small_cell_abundance_means.csv',index_col=0)
        assert spot.index.is_unique and spot.index.equals(abundance.index)
        ma=spot.candidate_abundance;epi=ma+spot.other_epithelial_abundance
        check=(spot.candidate_MLH1_reconstructed_log1p_cp10k*ma).sum()/epi.sum()
        np.testing.assert_allclose(check,data.mean_MLH1_epithelial.iloc[0],atol=1e-7)
        np.testing.assert_allclose(spot.candidate_fraction,ma/epi,atol=1e-7)
        np.testing.assert_allclose(spot.candidate_MLH1_reconstructed_log1p_cp10k,
            np.log1p(10000*spot.candidate_MLH1_allocated_counts/spot.candidate_library_allocated_counts),atol=1e-6)
        sid=data.index[0];patient=re.search(r'\((S\d+)_',directory.name).group(1)
        frames.append(data.assign(patientID=patient))
        z=classifier['scaler'].transform(data[features])
        contributions=z*classifier['model'].coef_[0]
        p=classifier.predict_proba(data[features])[0,1]
        records.append(dict(sampleID=sid,patientID=patient,score=p,
            ecology_only_score=ecology.predict_proba(data[features[:-1]])[0,1],
            candidate_fraction=metadata['candidate_fraction'],weighted_MLH1_proxy=check,
            candidate_MLH1_mean_proxy=metadata['candidate_MLH1_mean_proxy'],
            MLH1_standardized=float(z[0,-1]),MLH1_logit_contribution=float(contributions[0,-1]),
            n_spots=len(spot)))
        spotdata[sid]=spot
        for file in directory.glob('history_*.csv'):
            h=pd.read_csv(file,index_col=0).select_dtypes(include='number')
            if h.empty:continue
            v=h.iloc[:,-1].to_numpy();assert np.isfinite(v).all()
            convergence.append(dict(sampleID=sid,history=file.stem,epochs=len(v),
                first50_mean=np.mean(v[:50]),last50_mean=np.mean(v[-50:]),
                previous50_mean=np.mean(v[-100:-50]),
                last_window_relative_change=abs(np.mean(v[-50:])-np.mean(v[-100:-50]))/max(abs(np.mean(v[-100:-50])),1e-12)))
    # Labels are joined only after all features and model scores have been fixed.
    manifest=json.loads((FINAL/'data/stRNA/result/process/stRNA_cell2location_cell2location_manifest.json').read_text())
    labels={r['sampleID']:r['MMR'] for r in manifest['samples']}
    sections=pd.DataFrame(records);sections['MMR']=sections.sampleID.map(labels)
    assert sections.MMR.isin(['MSS','MSI']).all()
    sections['y']=sections.MMR.eq('MSI').astype(int)
    assert sections.groupby('patientID').y.nunique().eq(1).all()
    patients=sections.groupby('patientID',as_index=False).agg(MMR=('MMR','first'),y=('y','first'),
        score=('score','mean'),ecology_only_score=('ecology_only_score','mean'),
        weighted_MLH1_proxy=('weighted_MLH1_proxy','mean'),candidate_fraction=('candidate_fraction','mean'),
        n_sections=('sampleID','size'))
    assert len(patients)==7 and patients.n_sections.eq(2).all() and patients.y.sum()==2
    sections.to_csv(OUT/'section_predictions.csv',index=False);patients.to_csv(OUT/'patient_predictions.csv',index=False)
    spatial=pd.concat(frames);spatial.to_csv(OUT/'spatial_classifier_features.csv')
    audit=[]
    for f in features:
        lower,upper=train[f].min(),train[f].max()
        audit.append(dict(feature=f,training_min=lower,training_max=upper,spatial_min=spatial[f].min(),spatial_max=spatial[f].max(),
            n_sections_outside_training_range=int((spatial[f].lt(lower)|spatial[f].gt(upper)).sum())))
    pd.DataFrame(audit).to_csv(OUT/'feature_scale_audit.csv',index=False)
    pd.DataFrame(convergence).to_csv(OUT/'convergence_audit.csv',index=False)
    metrics=[]
    for level,table in [('Section',sections),('Patient',patients)]:
        for name,col in [('Weighted reconstructed transfer','score'),('18-subtype only transfer','ecology_only_score')]:
            metrics.append(dict(level=level,model=name,n=len(table),MSI=int(table.y.sum()),AUROC=roc_auc_score(table.y,table[col])))
        metrics.append(dict(level=level,model='Negative weighted MLH1 proxy',n=len(table),MSI=int(table.y.sum()),AUROC=roc_auc_score(table.y,-table.weighted_MLH1_proxy)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'descriptive_AUROC.csv',index=False)
    historical=pd.read_csv(FINAL/'model/reference/shared18_LDA_zscore2026/spatial_predictions.csv')
    comparison=[]
    for level,table,key in [('Section',sections,'sampleID'),('Patient',patients,'patientID')]:
        old=historical[historical.level.eq(level)].set_index(key)
        assert old.index.is_unique and set(old.index)==set(table[key])
        np.testing.assert_array_equal(old.loc[table[key],'y'],table.y)
        comparison.append(dict(level=level,historical_unweighted_AUROC=roc_auc_score(old.y,old.score),
            current_weighted_AUROC=roc_auc_score(table.y,table.score),
            interpretation='Reference reconstruction and classifier both changed; not an isolated weighting effect'))
    pd.DataFrame(comparison).to_csv(OUT/'historical_current_comparison.csv',index=False)
    font_manager.fontManager.addfont(subprocess.check_output(['fc-match','-f','%{file}','Arial'],text=True).strip())
    plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(13,9.933));grid=fig.add_gridspec(4,7,left=.12,right=.91,top=.94,bottom=.08,hspace=.48,wspace=.4,height_ratios=[1,1,1,1.4])
    v=np.concatenate([(x.candidate_MLH1_reconstructed_log1p_cp10k*x.candidate_fraction).to_numpy() for x in spotdata.values()])
    vmax=float(np.quantile(v,.99));axes=[]
    for i,r in enumerate(sections.itertuples()):
        ax=fig.add_subplot(grid[i//7,i%7]);spot=spotdata[r.sampleID]
        dots=ax.scatter(spot.x,spot.y,c=spot.candidate_MLH1_reconstructed_log1p_cp10k*spot.candidate_fraction,
            s=1.5,cmap='viridis',vmin=0,vmax=vmax,rasterized=False,linewidths=0)
        ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.set_title(f'{r.sampleID}\n{r.patientID}',fontsize=8);axes.append(ax)
    fig.text(.04,.985,'A  Spatial distribution of reconstructed, malignant-fraction weighted MLH1',fontsize=12,weight='bold',va='top')
    cb=fig.colorbar(dots,cax=fig.add_axes([.93,.58,.012,.30]));cb.set_label('Weighted reconstructed MLH1',fontsize=8)
    cb.solids.set_rasterized(False)
    ax=fig.add_subplot(grid[2,:]);x=np.arange(len(sections));colors=['#B65066' if y else '#277F8E' for y in sections.y]
    ax.scatter(x,sections.score,c=colors,s=30,label='New weighted transfer');ax.scatter(x,sections.ecology_only_score,c=colors,s=25,marker='x',label='18 subtypes only')
    ax.set_xticks(x,[s.replace('VISDS','') for s in sections.sampleID],rotation=45,ha='right',fontsize=8);ax.set_ylabel('MSI model score');ax.set_ylim(-.04,1.04);ax.set_title('B  Frozen-model section scores (exploratory)',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax=fig.add_subplot(grid[3,:3]);patient_metrics=metrics[metrics.level.eq('Patient')]
    ax.barh(range(3),patient_metrics.AUROC,color=['#277F8E','#8D9298','#B65066']);ax.set_yticks(range(3),['Weighted transfer','18 subtypes only','Weighted MLH1 proxy'],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,1.08);ax.set_xlabel('Descriptive patient AUROC')
    for i,v in enumerate(patient_metrics.AUROC):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_title('C  Inferred grouping: 7 patients, 2 MSI',loc='left',fontweight='bold')
    ax=fig.add_subplot(grid[3,4:]);ax.scatter(sections.candidate_fraction,sections.weighted_MLH1_proxy,c=colors,s=30);ax.set_xlabel('Estimated candidate / epithelial fraction');ax.set_ylabel('Section weighted MLH1 proxy');ax.set_title('D  Reconstructed inputs',loc='left',fontweight='bold')
    for ax in fig.axes[15:]:ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure04_weighted_spatial.{ext}',dpi=220)
    plt.close(fig)
    (OUT/'verification.json').write_text(json.dumps(dict(passed=True,n_sections=14,n_patients=7,
        official_expression_formula_verified=all(json.loads((d/'complete.json').read_text())['official_expression_formula_verified'] for d in directories),
        frozen_classifier_sha256=model_hash,
        labels_joined_after_predictions=True,feature_arithmetic_verified=True,
        measurement_equivalence_established=False,clinical_validation=False),indent=2))
    elbo=pd.DataFrame(convergence).query("history == 'history_elbo_train'")
    assert len(elbo)==14 and elbo.epochs.eq(500).all()
    m=next(x for x in audit if x['feature']=='mean_MLH1_epithelial')
    ref=json.loads((SOURCE/'reference/complete.json').read_text())
    readme=['# Frozen weighted scRNA classifier transfer to spatial data', '',
        'This is an exploratory technical transfer, not independent clinical validation.', '',
        '## Results', '',
        '| Level | Model | n | MSI | AUROC |', '| --- | --- | ---: | ---: | ---: |']
    readme.extend(f'| {r.level} | {r.model} | {r.n} | {r.MSI} | {r.AUROC:.6f} |' for r in metrics.itertuples())
    readme.extend(['', '## Provenance and interpretation', '',
        f'- Frozen classifier SHA-256: `{model_hash}`.',
        f'- Reference: {ref["n_cells"]} sampled GSE178341 tumor cells, {ref["n_genes"]} genes, 71 non-epithelial types plus two epithelial CNV states.',
        '- Other scored epithelial cells are not asserted to be biologically normal. Unscored epithelial cells were excluded.',
        '- Candidate malignant state comes from the corrected Stromal-only q95 CNV analysis; no spatial MSI labels were used for fitting.',
        '- Each reference type was sampled at most 1200 cells; all explicit seeds are 2026. Reference regression: 180 epochs; each section: 500 epochs, 200 posterior draws, N_cells_per_location=8, detection_alpha=20.',
        '- Reference per-epoch history was not retained in this completed fit. Reference convergence therefore cannot be established from a loss trace. All 14 section loss traces are saved.',
        f'- Section ELBO last50 versus previous50 relative change ranges from {elbo.last_window_relative_change.min():.4%} to {elbo.last_window_relative_change.max():.4%}; this is a diagnostic, not a convergence guarantee.',
        '- The candidate-type count reconstruction was checked against the installed official cell2location compute_expected_per_cell_type API in every section.',
        '- MLH1 is log1p(CP10K) in the reconstructed candidate-type spot profile, using the 12079 shared filtered genes. Section input = sum(MLH1 * candidate abundance) / sum(candidate + other epithelial abundance).',
        '- A log-transformed type-specific mixed-spot reconstruction is not the observed average single-cell log expression. Spatial normalization genes and platform differ; calibration and measurement equivalence remain unestablished.',
        '- The 18 subtype inputs are sums of inferred subtype abundances divided by the represented parent-type abundance. The reference cannot identify an unknown/omitted subtype component.',
        f'- Weighted MLH1 training range: {m["training_min"]:.8g} to {m["training_max"]:.8g}; spatial range: {m["spatial_min"]:.8g} to {m["spatial_max"]:.8g}; {m["n_sections_outside_training_range"]}/14 outside training range.',
        '- Scaler and joint classifier were never fitted on spatial data. The ecology-only comparator was fitted only on GSE178341 with the same scaler/LDA specification.',
        '- Labels are inferred from public sample grouping, with 7 patients and only 2 MSI. Two section probabilities are averaged per patient. All 14 sections, including the named low-quality replicate, were retained.',
        '- Historical/current comparison changes the reference, expression reconstruction and classifier together. It is not an isolated effect of the weighting formula.',
        '- PDFs/SVGs use vector points and text; PNGs are Word fallback images. Figure color clipping at the pooled 99th percentile is display-only.', '',
        '## Files', '',
        '- section_predictions.csv and patient_predictions.csv: continuous scores and grouping.',
        '- spatial_classifier_features.csv: exact 19 classifier inputs.',
        '- feature_scale_audit.csv and convergence_audit.csv: input and optimization diagnostics.',
        '- historical_current_comparison.csv: archived unweighted versus current weighted descriptive AUROCs.',
        '- Figure04_weighted_spatial.pdf/.svg/.png: spatial distributions, scores and descriptive AUROCs.',
        f'- Reconstruction artifacts: `{SOURCE}`.',
        '- Reconstruction entry point: model_code/run_weighted_spatial_transfer.py; summary: model_code/summarize_weighted_spatial_transfer.py.', '',
        'Official method references: https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_tutorial.html and https://cell2location.readthedocs.io/en/latest/cell2location.html.', ''])
    (OUT/'README.md').write_text('\n'.join(readme))
    print(metrics.to_string(index=False));print(sections.to_string(index=False));print(pd.DataFrame(audit).query('n_sections_outside_training_range > 0').to_string(index=False))

if __name__=='__main__':main()
