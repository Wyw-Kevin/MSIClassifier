"""Independent frozen TOSICA cell annotation only; never substitute legacy MLH1 proxies or fill unsupported subtype features."""
from pathlib import Path
import argparse,json,sys,random
import numpy as np,pandas as pd
from package_paths import project_root

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path)
    p.add_argument('--reference-family',required=True,choices=['historical_features','seed2026_cnv'])
    p.add_argument('--reference-dir',type=Path)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--dataset');p.add_argument('--h5ad',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--sample-col',default='Sample');p.add_argument('--tumor-col',default='Class')
    p.add_argument('--require-fixed18-coverage',action='store_true')
    args=p.parse_args();r=project_root(args.project_root);f=r/'最终模型';src=f/'scRNA_model';sys.path.insert(0,str(src))
    from scRNA_msi import preprocess as pp
    ref=args.reference_dir or src/('reference' if args.reference_family=='historical_features' else 'reference_seed2026')
    cfg=pp.model_set_config(ref);metadata=[];missing=[];labels=[]
    models={'Major':cfg['major'],**cfg['subtypes']}
    for name,c in models.items():
        for file in [c['weight'],'mask.npy','label_dictionary.csv']:
            if not (c['dir']/file).exists():missing.append(str(c['dir']/file))
        genes=pp.read_train_genes(c['dir']);classes=pp.read_label_dictionary(c['dir']);metadata.append({'model':name,'weight':str(c['dir']/c['weight']),'n_genes':len(genes),'n_labels':len(classes),'model_config':pp.read_model_config(c['dir'])})
        if name!='Major':labels.extend(classes)
    panel=pd.read_csv(f/'model/reference/selected_18_subtype_features.csv').feature.tolist()
    # This older selected-feature resource may represent a legacy panel; use the approved artifact's exact panel.
    import joblib
    panel=joblib.load(f/'model/reference/shared18_weighted_mlh1_2026/model.joblib')['features'][:-1]
    unsupported=[s for s in panel if s not in labels]
    report={'reference_family':args.reference_family,'fixed18_supported':18-len(unsupported),'fixed18_unsupported':unsupported,'models':metadata,'missing_checkpoint_files':missing,'output_scope':'cell annotation only; no feature imputation, classifier score or MLH1 proxy'}
    if args.check_only:
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if args.require_fixed18_coverage and unsupported:raise ValueError('Unmeasured fixed18 subtypes: '+', '.join(unsupported))
    if missing:raise FileNotFoundError('Provide exact frozen checkpoints before annotation: '+', '.join(missing))
    if not all([args.h5ad,args.dataset,args.output_dir]):raise ValueError('--h5ad, --dataset and --output-dir are required for annotation')
    import torch
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2026)
    device=pp.choose_device(args.device);adata=pp.read_h5ad_dataset(args.h5ad,h5ad_backed='r',use_raw=False)
    try:
        cells=pp.read_obs_table(adata,adata.n_obs,sample_col=args.sample_col,tumor_col=args.tumor_col)
        counts=cells.groupby('Sample',sort=False).size();keep=np.flatnonzero(cells.Sample.isin(counts[counts>=200].index).to_numpy())
        if len(keep)!=len(cells):
            filtered=adata[keep].copy();adata.file.close();adata=filtered;cells=pp.read_obs_table(adata,adata.n_obs,sample_col=args.sample_col,tumor_col=args.tumor_col)
        rows=np.arange(adata.n_obs,dtype=np.int64);vendor=src/'reference/TOSICA_source'
        major,prob,_,_,info=pp.predict_model(adata,rows,cfg['major']['dir'],cfg['major']['weight'],args.batch_size,device,'Major',tosica_source=vendor)
        cells['major_prediction']=major;cells['major_probability']=prob
        for col in ['subtype_model','subtype_prediction','final_migrated_subtype']:cells[col]=''
        cells['subtype_probability']=np.nan;infos=[info]
        for name,c in cfg['subtypes'].items():
            ix=np.flatnonzero(cells.major_prediction.eq(name).to_numpy())
            if not len(ix):continue
            label,prob,_,_,info=pp.predict_model(adata,ix,c['dir'],c['weight'],args.batch_size,device,name,tosica_source=vendor)
            cells.loc[ix,'subtype_model']=name;cells.loc[ix,'subtype_prediction']=label;cells.loc[ix,'final_migrated_subtype']=label;cells.loc[ix,'subtype_probability']=prob;infos.append(info)
        args.output_dir.mkdir(parents=True,exist_ok=True);cells.to_csv(args.output_dir/f'{args.dataset}_tosica_migration_cells.csv',index=False)
        report.update({'dataset':args.dataset,'n_cells':len(cells),'seed':2026,'model_infos':infos,'sample_QC':'total cells >=200; cohort processed separately'})
        (args.output_dir/'annotation_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        if hasattr(adata,'file'):adata.file.close()
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
