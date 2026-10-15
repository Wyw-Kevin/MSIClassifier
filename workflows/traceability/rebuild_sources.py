"""Trace actual analysis IDs/labels and verify renamed GEO columns by expression."""
from pathlib import Path
import csv,gzip,io,json,re,hashlib
import numpy as np,pandas as pd
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root',required=True,type=Path,help='Archived original raw-data project layout')
parser.add_argument('--output-dir',required=True,type=Path)
args=parser.parse_args()
ROOT=args.project_root.resolve()
REPO=Path(__file__).resolve().parents[2]
TARGET=args.output_dir.resolve();TARGET.mkdir(parents=True,exist_ok=True)
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def open_text(path):
    with path.open('rb') as h:magic=h.read(2)
    return gzip.open(path,'rt') if magic==b'\x1f\x8b' else path.open()
def matrix_parts(path,table=False):
    meta={};lines=[];go=False
    with open_text(path) as h:
        for line in h:
            if line.startswith('!series_matrix_table_end'):break
            if go:
                if table:lines.append(line)
                else:break
            elif line.startswith('!series_matrix_table_begin'):go=True
            elif line.startswith('!Sample_'):
                fields=next(csv.reader([line],delimiter='\t'));meta.setdefault(fields[0],[]).append(fields[1:])
    return meta,pd.read_csv(io.StringIO(''.join(lines)),sep='\t',index_col=0) if table else None
def status_fields(meta,pos):
    chars=[row[pos] for row in meta.get('!Sample_characteristics_ch1',[]) if len(row)>pos]
    return ' | '.join(v for v in chars if re.search(r'(microsatellite|\bmsi[ _-]?(status|phenotype)?\b|mmr\.status|mismatch repair)',v,re.I))

pred=pd.read_csv(REPO/'workflows/bulk/classifier_snapshot/predictions.csv')
bulk=pred[pred.panel.eq('panel12')&pred.split.isin(['development_OOF','retrospective'])].drop_duplicates(['dataset','sampleID'])
assert len(bulk)==1212
rows=[];renamed=[];audits=[];fingerprints=[]
for ds,part in bulk.groupby('dataset',sort=False):
    folder=ROOT/'bulk_data'/ds;clinical_path=folder/'clinical.csv';expression_path=folder/'expression.csv'
    clinical=pd.read_csv(clinical_path).set_index('SampleID');columns=pd.read_csv(expression_path,nrows=0).columns.tolist()[1:]
    assert part.sampleID.isin(clinical.index).all() and part.sampleID.isin(columns).all()
    rawpath=folder/'raw'/(ds+'_series_matrix.txt.gz');meta={};mapping={};similarity={};rawvalues=None;local=None
    if rawpath.exists():meta,rawvalues=matrix_parts(rawpath,ds in ['GSE27544','GSE75316'])
    public=meta.get('!Sample_geo_accession',[[]])[0];titles=meta.get('!Sample_title',[[]])[0]
    if ds in ['GSE27544','GSE75316']:
        annotation=ROOT/('data/GSE27544/GPL13158.soft.gz' if ds=='GSE27544' else 'bulk_data/GSE39582/raw/GPL570.soft.gz')
        with open_text(annotation) as h:
            lines=[];go=False
            for line in h:
                if line.startswith('!platform_table_end'):break
                if go:lines.append(line)
                if line.startswith('!platform_table_begin'):go=True
        ann=pd.read_csv(io.StringIO(''.join(lines)),sep='\t',dtype=str,low_memory=False).set_index('ID')
        symbol=ann['Gene Symbol'].dropna();symbol=symbol[~symbol.str.contains('///',regex=False)]
        probes=rawvalues.index.intersection(symbol.index);raw=rawvalues.loc[probes].copy();raw.index=symbol.loc[probes]
        means=raw.groupby(level=0).mean();local=pd.read_csv(expression_path,index_col=0)
        genes=means.index.intersection(local.index);x=means.loc[genes].to_numpy();y=local.loc[genes].to_numpy()
        counts=np.stack([np.isclose(x,y[:,j,None],atol=1e-7,rtol=0).sum(axis=0) for j in range(y.shape[1])])
        best=counts.argmax(axis=1);second=np.sort(counts,axis=1)[:,-2]
        assert len(set(best))==len(best) and np.all(counts.max(axis=1)>.75*len(genes)) and np.all(counts.max(axis=1)>second*10)
        for j,col in enumerate(local.columns):
            idx=int(best[j]);accession=means.columns[idx];assert accession==public[idx]
            mapping[col]=idx;similarity[col]=(int(counts[j,idx]),int(second[j]))
            renamed.append({'dataset':ds,'analysis_sampleID':col,'stored_matrix_column':col,'public_GSM':accession,'public_sample_title':titles[idx],
                'mapping_basis':'Unique maximum exact expression-value agreement across mapped raw probe-mean genes; independently checked against all source columns',
                'common_genes_compared':len(genes),'matching_gene_values':int(counts[j,idx]),'next_best_matching_values':int(second[j]),
                'raw_series_matrix':str(rawpath.relative_to(ROOT)),'raw_sha256':SHA(rawpath),'annotation_sha256':SHA(annotation)})
            # Retain a small independent-value witness for replay; not all preprocessing aliases need match.
            matching=np.flatnonzero(np.isclose(x[:,idx],y[:,j],atol=1e-7,rtol=0))[:128]
            for k in matching:fingerprints.append({'dataset':ds,'analysis_sampleID':col,'public_GSM':accession,'gene':genes[k],'raw_probe_mean':x[k,idx],'stored_gene_expression':y[k,j]})
        audits.append({'dataset':ds,'n_renamed':len(mapping),'common_genes':len(genes),'min_matching_fraction':float(counts.max(axis=1).min()/len(genes)),
            'max_next_best_matching_fraction':float(second.max()/len(genes)),'unique_bijective_expression_mapping':True})
    for row in part.itertuples():
        c=clinical.loc[row.sampleID];label='MSI' if row.y else 'MSS';assert c.MMR==label
        if row.sampleID in mapping:idx=mapping[row.sampleID]
        else:idx=public.index(row.sampleID) if row.sampleID in public else None
        accession=public[idx] if idx is not None else (row.sampleID if row.sampleID.startswith('GSM') else '')
        source_value=status_fields(meta,idx) if idx is not None else str(c.get('source_value',''))
        if not source_value:source_value=str(c.get('source_value',''))
        msil=bool(re.search(r'\bMSI[-_ ]?L\b',source_value,re.I))
        rows.append({'modality':'bulk','dataset':ds,'analysis_sampleID':row.sampleID,'patient_or_groupID':row.sampleID[:12] if ds.startswith('TCGA') else '',
            'original_sample_identifier':accession or row.sampleID,'stored_matrix_column':row.sampleID,'public_sample_title':titles[idx] if idx is not None else '',
            'analysis_label':label,'y':int(row.y),'source_label_field':str(c.get('raw_msi_field',c.get('source_field',''))),'source_label_value':source_value,
            'source_file':str(clinical_path.relative_to(ROOT)),'source_sha256':SHA(clinical_path),'raw_matrix_source':str(rawpath.relative_to(ROOT)) if rawpath.exists() else '',
            'mapping_status':'Expression-verified renamed column' if row.sampleID in mapping else 'Exact stored/public identifier',
            'role':'Subtype development' if ds in ['GSE39582','TCGA-COADREAD'] else 'Bulk retrospective evaluation',
            'assay_confirmation':'Original assay not individually verified from stored metadata','included_primary':True,
            'exception_note':'Raw MSI-L stored as negative in current archived analysis' if msil else ''})

ledger_path=REPO/'workflows/scRNA/primary17/inputs/patient_ledger.csv';ledger=pd.read_csv(ledger_path)
for ds in ['GSE178341','GSE132465']:
    f=pd.read_csv(REPO/'workflows/scRNA/primary17/tables'/(ds+'_primary17_features.csv'))
    l=ledger[ledger.dataset.eq(ds)].set_index('sampleID').loc[f.sampleID]
    for sample,datum in zip(f.itertuples(),l.itertuples()):
        assert sample.y==datum.y
        original=datum.sample_source if isinstance(datum.sample_source,str) else sample.sampleID
        rows.append({'modality':'scRNA','dataset':ds,'analysis_sampleID':sample.sampleID,'patient_or_groupID':datum.patientID,'original_sample_identifier':original,
            'stored_matrix_column':sample.sampleID,'public_sample_title':'','analysis_label':'MSI' if sample.y else 'MSS','y':int(sample.y),
            'source_label_field':'MMR_IHC/MMRStatus' if ds=='GSE178341' else 'Public patient MSI annotation','source_label_value':datum.MMR_source,
            'source_file':str(ledger_path.relative_to(REPO)),'source_sha256':SHA(ledger_path),'raw_matrix_source':ds+'.h5ad obs',
            'mapping_status':'Archived patient/sample crosswalk; predictions and labels agree','role':'scRNA development' if ds=='GSE178341' else 'scRNA external evaluation',
            'assay_confirmation':'Public IHC-derived MMR record' if ds=='GSE178341' else 'Original assay not individually verified','included_primary':True,'exception_note':''})

clinpath=ROOT/'最终模型/data/scRNA/result/diagnostics/GSE200997_locked_shared18/clinical_labels.csv'
cl=pd.read_csv(clinpath);qcpath=clinpath.parent/'sample_cell_QC.csv';qc=pd.read_csv(qcpath).set_index('sampleID')
original=ROOT/'最终模型/data/scRNA/external/GSE200997/clinical_original.xlsx';orig=pd.read_excel(original,header=None)
source_ids=orig.iloc[:,0].astype(str);MSI=orig.iloc[:,4].astype(str)
f=pd.read_csv(REPO/'workflows/scRNA/primary17/tables/GSE200997_primary17_features.csv').set_index('sampleID')
for row in cl.itertuples():
    # Published table uses T_CAC identifiers; preserve the tumor prefix and normalize case only.
    base=row.sampleID.removeprefix('T_');matches=np.flatnonzero(source_ids.str.lower().str.replace(' ','').eq(row.sampleID.lower()) & MSI.isin(['MSS','MSI-H']))
    assert len(matches)==1,(row.sampleID,len(matches));source_label=MSI.iloc[matches[0]];value=source_ids.iloc[matches[0]]
    assert (('MSI' in source_label.upper()) and ('MSS' not in source_label.upper()))==bool(row.y)
    included=row.sampleID in f.index
    if included:assert row.y==f.loc[row.sampleID,'y']
    else:assert row.sampleID=='T_cac5' and qc.loc[row.sampleID,'total_cells']==174
    rows.append({'modality':'scRNA','dataset':'GSE200997','analysis_sampleID':row.sampleID,'patient_or_groupID':base,'original_sample_identifier':value,'stored_matrix_column':row.sampleID,
      'public_sample_title':'','analysis_label':'MSI' if row.y else 'MSS','y':int(row.y),'source_label_field':'Supplementary Table1 MSI Status','source_label_value':source_label,
      'source_file':str(original.relative_to(ROOT)),'source_sha256':SHA(original),'raw_matrix_source':'GSE200997.h5ad obs Sample','mapping_status':'Case-normalized exact published table identifier; tumor prefix retained',
      'role':'scRNA external evaluation' if included else 'QC excluded; not scored','assay_confirmation':'Original assay not individually verified','included_primary':included,
      'exception_note':'' if included else '174 total cells <200; no current-model prediction'})

spatialpath=REPO/'workflows/scRNA/primary17/spatial/section_predictions.csv';sp=pd.read_csv(spatialpath)
for row in sp.itertuples():
    rows.append({'modality':'spatial','dataset':'VISDS','analysis_sampleID':row.sampleID,'patient_or_groupID':row.patientID,'original_sample_identifier':row.sampleID,'stored_matrix_column':row.sampleID,
      'public_sample_title':'','analysis_label':row.MMR,'y':int(row.y),'source_label_field':'Public grouping inference','source_label_value':row.MMR,
      'source_file':str(spatialpath.relative_to(REPO)),'source_sha256':SHA(spatialpath),'raw_matrix_source':'Saved section data and analysis_plan.json','mapping_status':'Exact section/patient mapping',
      'role':'Technical proxy-transfer evaluation','assay_confirmation':'No patient-level independent molecular assay confirmation','included_primary':True,'exception_note':'Inferred grouping only; not gold-standard MSI status'})
full=pd.DataFrame(rows);assert not full.duplicated(['dataset','analysis_sampleID']).any()
assert len(full)==1329 and full.included_primary.sum()==1328
for column in full.select_dtypes(include='object'):
    full[column]=full[column].str.replace('最终模型/','final_model/',regex=False)
full.to_csv(TARGET/'S5_sample_label_sources.csv',index=False)
pd.DataFrame(renamed).to_csv(TARGET/'S5_renamed_GEO_ID_mapping.csv',index=False)
pd.DataFrame(fingerprints).to_csv(TARGET/'GEO_ID_expression_fingerprints.csv.gz',index=False,compression='gzip')
full[full.exception_note.str.contains('MSI-L')].to_csv(TARGET/'archived_MSI_L_negative_samples.csv',index=False)
proof={'passed':True,'all_analyzed_samples':1328,'excluded_QC_sample':1,'rows_by_modality':full.groupby('modality').size().to_dict(),'renamed_GEO_samples':len(renamed),
    'expression_mapping_audits':audits,'MSI_L_archived_negative_count':int(full.exception_note.str.contains('MSI-L').sum()),'labels_changed':False,'assay_gaps_not_inferred':True}
(TARGET/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
