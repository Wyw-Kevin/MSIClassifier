"""Score complete already measured fixed18+weightedMLH1 features with the frozen model."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
from package_paths import artifact_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path)
    p.add_argument('--feature-table',required=True,type=Path)
    p.add_argument('--weighted-mlh1-column',default='weighted_MLH1',help='Must contain mu*N_malignant/N_epithelial from the same retained CNV cells.')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    a=joblib.load(artifact_path(args.project_root));features=a['features']
    if a['seed']!=2026 or a['formula']!='mean_MLH1 * N_malignant / N_epi':raise ValueError('Wrong frozen model identity')
    t=pd.read_csv(args.feature_table,float_precision='round_trip')
    if args.weighted_mlh1_column not in t:raise ValueError('Missing explicitly weighted MLH1 column; no proxy substitution')
    t=t.copy();t[features[-1]]=t[args.weighted_mlh1_column]
    missing=[f for f in features if f not in t]
    if missing:raise ValueError('Missing/unmeasured fixed18 features: '+', '.join(missing))
    X=t[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():raise ValueError('Missing/unmeasured or non-finite inputs; no imputation or zero filling')
    if len(t)==0 or 'sampleID' not in t or t.sampleID.duplicated().any():raise ValueError('Require unique sampleID rows')
    result=t[[c for c in ['dataset','sampleID','MMR'] if c in t]].copy()
    result['score']=a['pipeline'].predict_proba(t[features])[:,1]
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(json.dumps({'n':len(result),'model':str(artifact_path(args.project_root)),'formula':a['formula'],'refitting':False,'output':str(args.output)}))
if __name__=='__main__':main()
