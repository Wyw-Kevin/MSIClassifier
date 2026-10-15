"""Current17-panel evaluation conditional on recorded source screening/CNV inputs."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

HERE = Path(__file__).resolve().parents[1]
I, M, T = HERE/"inputs", HERE/"models", HERE/"tables"
DS = ["GSE178341", "GSE132465", "GSE200997"]
for folder in [M,T]: folder.mkdir(exist_ok=True)
source_hash = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in I.iterdir() if p.is_file()}
data = {ds:pd.read_csv(I/(ds+".csv"), float_precision="round_trip") for ds in DS}
qfeatures = pd.read_csv(I/"cnv_quantiles.csv", float_precision="round_trip")
source = joblib.load(I/"archived17_foldfull.joblib")
cols = source["features"]; panel = cols[:-1]
assert len(panel)==17 and cols[-1]=="weighted_MLH1" and source["seed"]==2026
membership=pd.read_csv(I/"threshold_membership.csv")
actual=membership[membership.screening_cutoff.eq(.7)&membership.in_model]
assert actual.feature.tolist()==panel and actual.reference_oriented_AUROC.gt(.7).all()
patient_ledger=pd.read_csv(I/"patient_ledger.csv")
for ds,frame in data.items():
    q=qfeatures[qfeatures.dataset.eq(ds)&qfeatures["quantile"].eq(.95)].set_index("sampleID").loc[frame.sampleID]
    frame["mu"]=q.mu.to_numpy();frame["p"]=q.p.to_numpy()
    assert np.allclose(frame.weighted_MLH1,q.W,atol=2e-12,rtol=0)
    frame["n_candidate"]=q.n_candidate.to_numpy();frame["n_epithelial"]=q.n_epithelial.to_numpy()
    if "patientID" not in frame:frame["patientID"]=frame.sampleID
    frame["patientID"]=frame.patientID.fillna(frame.sampleID).astype(str)
    pm=patient_ledger[patient_ledger.dataset.eq(ds)].set_index("sampleID").patientID
    frame["patientID"]=frame.sampleID.map(pm).fillna(frame.patientID).astype(str)
    frame.to_csv(T/(ds+"_primary17_features.csv"),index=False)
dev=data[DS[0]]
assert len(dev)==64 and dev.patientID.nunique()==62 and dev.groupby("patientID").fold.nunique().eq(1).all()

names={"Joint":"17 subtypes + weighted MLH1","Unweighted":"17 subtypes + mean MLH1",
       "Subtype":"17 subtypes only","MLH1":"Weighted MLH1 only","Proportion":"Candidate fraction only",
       "SubtypeProportion":"17 subtypes + candidate fraction","Separate":"17 subtypes + mean MLH1 + candidate fraction"}
config={"Joint":cols,"Unweighted":panel+["mu"],"Subtype":panel,"MLH1":["weighted_MLH1"],
        "Proportion":["p"],"SubtypeProportion":panel+["p"],"Separate":panel+["mu","p"]}
predictions=[];fitted={};fit_records=[]
def rows(pipe,frame,features,model,split,fold):
    score=pipe.predict_proba(frame[features].to_numpy())[:,1]
    return [dict(model=model,dataset=frame.dataset.iloc[0],split=split,sampleID=s,y=int(y),score=float(p),
                 patientID=pid,fold=str(fold)) for s,y,p,pid in zip(frame.sampleID,frame.y,score,frame.patientID)]
def fit(frame,features,name,fold):
    pipe=clone(source["pipeline"]).fit(frame[features].to_numpy(),frame.y.to_numpy())
    artifact={"pipeline":pipe,"features":features,"seed":2026,"fold":str(fold),
              "training_dataset":DS[0],"training_samples":frame.sampleID.tolist(),"training_patients":frame.patientID.tolist(),
              "variant":name,"target":"MSI","formula":"mean_MLH1 * N_malignant / N_epi" if name=="Joint" else "control",
              "screening_cutoff":.70,"n_subtypes":17,"selection":"All source orientedAUROC>0.70; no count cap",
              "selection_basis":"Highest development groupedOOF AUROC among0.60/0.65/0.70/0.75; chosen after threshold comparison"}
    path=M/f"{name}_fold{fold}.joblib";joblib.dump(artifact,path)
    fit_records.append({"model":name,"fold":str(fold),"features":";".join(features),"path":str(path.relative_to(HERE)),
                        "n_training_samples":len(frame),"n_training_patients":frame.patientID.nunique(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return pipe

for name,features in config.items():
    fitted[name]={}
    for fold in range(1,6):
        train=dev[dev.fold.ne(fold)];test=dev[dev.fold.eq(fold)]
        assert not set(train.patientID)&set(test.patientID)
        pipe=fit(train,features,name,fold);fitted[name][fold]=pipe
        predictions+=rows(pipe,test,features,name,"grouped_OOF",fold)
    pipe=fit(dev,features,name,"full");fitted[name]["full"]=pipe
    for ds,frame in data.items():predictions+=rows(pipe,frame,features,name,"training_apparent" if ds==DS[0] else "external","full")
pred=pd.DataFrame(predictions);pred.to_csv(T/"all_predictions.csv",index=False)
# Independently re-fit primary must reproduce the actual archived17 checkpoints.
errors=[]
for fold in [1,2,3,4,5,"full"]:
    archived=joblib.load(I/f"archived17_fold{fold}.joblib")
    frame=dev[dev.fold.eq(fold)] if fold!="full" else dev
    old=archived["pipeline"].predict_proba(frame[cols].to_numpy())[:,1]
    new=fitted["Joint"][fold].predict_proba(frame[cols].to_numpy())[:,1]
    errors.append(float(np.max(np.abs(old-new))))
assert max(errors)<1e-12

# One seed2026 stream, advancing across cohorts; reuse and verify saved patient draws where available.
saved=np.load(I/"original_patient_bootstrap.npz",allow_pickle=False)
rng=np.random.default_rng(2026);draw_arrays={};boot_counts={}
for ds in DS:
    frame=data[ds].sort_values("sampleID").reset_index(drop=True)
    ids=pd.Index(saved[ds+"_patients"]) if ds+"_patients" in saved else pd.Index(frame.patientID.unique())
    enc=ids.get_indexer(frame.patientID);assert (enc>=0).all()
    py=np.array([np.unique(frame.y.to_numpy()[enc==j]).item() for j in range(len(ids))])
    positive=np.flatnonzero(py==1);negative=np.flatnonzero(py==0)
    draws=np.array([np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)] for _ in range(2000)])
    if ds in saved:assert np.array_equal(draws,saved[ds])
    counts=np.array([np.bincount(draw,minlength=len(ids))[enc] for draw in draws])
    boot_counts[ds]=(frame.sampleID.tolist(),counts)
    draw_arrays[ds]=draws;draw_arrays[ds+"_patients"]=ids.to_numpy(dtype=str)
np.savez_compressed(T/"patient_bootstrap_indices.npz",**draw_arrays)
metrics=[];deltas=[]
for (ds,split),part in pred.groupby(["dataset","split"],sort=False):
    sampleids,counts=boot_counts[ds]
    wide=part.pivot(index="sampleID",columns="model",values="score").reindex(sampleids)
    y=part.drop_duplicates("sampleID").set_index("sampleID").y.reindex(sampleids).to_numpy()
    values=wide.to_numpy().T;pos=values[:,y==1];neg=values[:,y==0]
    kernel=(pos[:,:,None]>neg[:,None,:]).astype(float)+.5*(pos[:,:,None]==neg[:,None,:])
    pw,nw=counts[:,y==1],counts[:,y==0]
    boot=np.einsum("bi,mij,bj->bm",pw,kernel,nw)/(pw.sum(axis=1)*nw.sum(axis=1))[:,None]
    bs=pd.DataFrame(boot,columns=wide.columns);points={}
    for name in wide.columns:
        s=wide[name].to_numpy();auc=roc_auc_score(y,s);points[name]=auc;lo,hi=np.quantile(bs[name],[.025,.975])
        metrics.append(dict(dataset=ds,split=split,model=name,model_label=names[name],n=len(wide),positive=int(y.sum()),
                            AUROC=auc,CI_low=lo,CI_high=hi,BA=balanced_accuracy_score(y,s>=.5),Brier=brier_score_loss(y,s)))
    for baseline in ["Unweighted","Subtype","MLH1"]:
        delta=points["Joint"]-points[baseline];reps=(bs.Joint-bs[baseline]).to_numpy();lo,hi=np.quantile(reps,[.025,.975])
        pv=float((1+np.count_nonzero(np.abs(reps-delta)>=abs(delta)-1e-12))/2001) if np.ptp(reps)>1e-12 else np.nan
        deltas.append(dict(dataset=ds,split=split,baseline=baseline,baseline_label=names[baseline],delta=delta,CI_low=lo,CI_high=hi,p_two_sided_null_centered=pv))
pd.DataFrame(metrics).to_csv(T/"metrics.csv",index=False)
pd.DataFrame(deltas).to_csv(T/"paired_differences.csv",index=False)

# Recompute CNV sensitivity for current17, including both refits and frozenq95 perturbations.
cnv_predictions=[]
for quantile in [.90,.95,.99]:
    frames={}
    for ds,frame in data.items():
        frame=frame.copy();q=qfeatures[qfeatures.dataset.eq(ds)&qfeatures["quantile"].eq(quantile)].set_index("sampleID").loc[frame.sampleID]
        assert q.scoreable.all() and q.n_candidate.gt(0).all() and q.n_epithelial.gt(0).all()
        frame["weighted_MLH1"]=q.W.to_numpy();frames[ds]=frame
    pipes={}
    for fold in [1,2,3,4,5,"full"]:
        train=frames[DS[0]] if fold=="full" else frames[DS[0]][frames[DS[0]].fold.ne(fold)]
        pipe=fitted["Joint"][fold] if quantile==.95 else fit(train,cols,f"CNVq{int(quantile*100)}",fold)
        pipes[fold]=pipe
    for mode in ["refit","frozen_q95"]:
        use=pipes if mode=="refit" else fitted["Joint"]
        for fold in range(1,6):
            frame=frames[DS[0]][frames[DS[0]].fold.eq(fold)]
            cnv_predictions.extend([dict(quantile=quantile,mode=mode,**r) for r in rows(use[fold],frame,cols,"Joint","grouped_OOF",fold)])
        for ds,frame in frames.items():
            cnv_predictions.extend([dict(quantile=quantile,mode=mode,**r) for r in rows(use["full"],frame,cols,"Joint","training_apparent" if ds==DS[0] else "external","full")])
cp=pd.DataFrame(cnv_predictions);cp.to_csv(T/"CNV_sensitivity_predictions.csv",index=False)
cm=[]
for (q,mode,ds,split),part in cp.groupby(["quantile","mode","dataset","split"]):
    cm.append(dict(quantile=q,mode=mode,dataset=ds,split=split,n=len(part),AUROC=roc_auc_score(part.y,part.score),
                   BA=balanced_accuracy_score(part.y,part.score>=.5),Brier=brier_score_loss(part.y,part.score)))
pd.DataFrame(cm).to_csv(T/"CNV_sensitivity_metrics.csv",index=False)

# Current low-candidate-count diagnostic; it is a subset of frozen current predictions.
low=[]
for ds in DS:
    omit=data[ds].loc[data[ds].n_candidate.lt(10),"sampleID"].tolist()
    for name in ["Joint","Unweighted","Subtype","MLH1"]:
        z=pred[pred.dataset.eq(ds)&pred.split.eq("grouped_OOF" if ds==DS[0] else "external")&pred.model.eq(name)&~pred.sampleID.isin(omit)]
        low.append(dict(dataset=ds,model=names[name],omitted=";".join(omit),n=len(z),AUROC=roc_auc_score(z.y,z.score),interpretation="Subset of unchanged current17 predictions; no refit; <10 is not an eligibility rule"))
pd.DataFrame(low).to_csv(T/"low_candidate_count_sensitivity.csv",index=False)
pd.DataFrame(fit_records).to_csv(T/"model_index.csv",index=False)
pd.DataFrame({"feature":panel}).to_csv(T/"selected17_features.csv",index=False)
threshold=pd.read_csv(I/"threshold_performance.csv");threshold=threshold[threshold.family.eq("subtype_threshold")]
threshold.to_csv(T/"threshold_performance.csv",index=False)
devscreen=threshold[threshold.dataset.eq(DS[0])]
assert devscreen.loc[devscreen.AUROC.idxmax(),"screening_cutoff"]==.7
for ds in DS:
    match=pd.DataFrame(metrics).query("model=='Joint' and split!='training_apparent' and dataset==@ds").iloc[0]
    archived=threshold[threshold.dataset.eq(ds)&threshold.screening_cutoff.eq(.7)].iloc[0]
    assert abs(match.AUROC-archived.AUROC)<1e-12
assert all(hashlib.sha256((I/name).read_bytes()).hexdigest()==value for name,value in source_hash.items())
report={"status":"computed and internally checked","primary_subtypes":panel,"n_subtypes":17,"cutoff":.70,
        "max_fresh_vs_archived17_score_error":max(errors),"threshold_selected_by":"Maximum GSE178341 groupedOOF among four compared cutoffs; final choice made after inspecting development results",
        "external_performance_used_for_selection":False,"development_OOF_used_for_cutoff_selection":True,
        "confidence_intervals":"2000 stratified patient-cluster draws; fixed predictions; no upstream or cutoff-selection uncertainty",
        "all_explicit_seeds":2026,"old_weights_overwritten":False,"input_sha256":source_hash}
(HERE/"computation_verification.json").write_text(json.dumps(report,indent=2))
print(pd.DataFrame(metrics).query("split!='training_apparent'")[["dataset","model","AUROC","CI_low","CI_high"]].to_string(index=False))
print(pd.DataFrame(deltas).query("split!='training_apparent'").to_string(index=False))
print(pd.DataFrame(cm).query("split!='training_apparent'").to_string(index=False))
