"""Current paired spatial display; independent of manuscript document editing."""
from pathlib import Path
import numpy as np,pandas as pd,joblib,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'resources/transfer'
CURRENT=HERE/'resources/report'
OUT=HERE/'outputs/figures'

def spatial():
    font_manager.fontManager.addfont(subprocess.check_output(['fc-match','-f','%{file}','Arial'],text=True).strip())
    plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':False})
    model=joblib.load(CURRENT/'ecology_only_comparator.joblib')
    groups=pd.read_csv(SOURCE/'reference/mapping.csv').set_index('small_type').major_type
    sections=pd.read_csv(CURRENT/'section_predictions.csv').sort_values('sampleID')
    metrics=pd.read_csv(CURRENT/'descriptive_AUROC.csv')
    data={};records=[]
    for row in sections.itertuples():
        folder=next(SOURCE.glob(row.sampleID+'(*'))
        s=pd.read_csv(folder/'spot_mlh1_and_abundance.csv.gz',index_col=0)
        a=pd.read_csv(folder/'small_cell_abundance_means.csv',index_col=0)
        assert s.index.equals(a.index)
        epi=a.Epi_CNV_candidate+a.Epi_other
        fraction=epi/a.sum(axis=1)
        mask=fraction>=fraction.quantile(.75)
        x=pd.DataFrame(index=a.index)
        for f in model['features']:
            denom=a.loc[:,groups[groups.eq(groups.loc[f])].index].sum(axis=1)
            assert denom.gt(0).all()
            x[f]=a[f]/denom
        s['local_ecology_score']=model['pipeline'].predict_proba(x[model['features']].to_numpy())[:,1]
        s['weighted_MLH1_display']=s.candidate_MLH1_reconstructed_log1p_cp10k*s.candidate_fraction
        s['epithelial_fraction']=fraction
        s['epithelial_enriched_display']=mask
        data[row.sampleID]=s
        records.append(s.assign(sampleID=row.sampleID))
    pd.concat(records).to_csv(OUT/'Figure04_spot_display_values.csv.gz',compression='gzip',index_label='spot_id')
    vmax=float(np.quantile(np.concatenate([d.weighted_MLH1_display for d in data.values()]),.99))
    fig=plt.figure(figsize=(16,12.225))
    grid=fig.add_gridspec(7,4,left=.025,right=.55,bottom=.05,top=.94,wspace=.18,hspace=.33)
    for i,row in enumerate(sections.itertuples()):
        s=data[row.sampleID];m=s.epithelial_enriched_display.to_numpy()
        for j in range(2):
            ax=fig.add_subplot(grid[i//2,(i%2)*2+j]);ax.scatter(s.x,s.y,s=2,c='#E9EDF1',linewidths=0,rasterized=False)
            use=m if j==0 else ~m
            values=s.weighted_MLH1_display if j==0 else s.local_ecology_score
            scatter=ax.scatter(s.x[use],s.y[use],c=values[use],s=3,linewidths=0,
                cmap='Reds' if j==0 else 'RdYlBu_r',vmin=0,vmax=vmax if j==0 else 1,rasterized=False)
            if j==0:mlh1_mappable=scatter
            else:ecology_mappable=scatter
            ax.set_aspect('equal');ax.invert_yaxis();ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('#D6DDE3');spine.set_linewidth(.45)
            label='Epithelial-enriched MLH1' if j==0 else '17-subtype score'
            ax.set_title(f'{row.sampleID} | {row.MMR}\n{label}',fontsize=6.6,pad=3)
    fig.text(.025,.978,'A  Spatial MLH1 expression and 17-subtype patterns',fontsize=12,weight='bold')
    for mapper,y,label in [(mlh1_mappable,.56,'Weighted reconstructed MLH1'),(ecology_mappable,.11,'Local 17-subtype score')]:
        cb=fig.colorbar(mapper,cax=fig.add_axes([.572,y,.010,.31]));cb.set_label(label,fontsize=8);cb.ax.tick_params(labelsize=7);cb.solids.set_rasterized(False)
    ax=fig.add_axes([.71,.41,.265,.47]);ordered=sections.sort_values('score',ascending=False);y=np.arange(len(ordered))
    for offset,column,color,label in [(-.23,'weighted_MLH1_proxy','#CA6575','Weighted MLH1'),(0,'ecology_only_score','#4F95BE','17-subtype ecology'),(.23,'score','#43A394','Combined model')]:
        ax.barh(y+offset,ordered[column],height=.21,color=color,label=label)
        for yy,v in zip(y+offset,ordered[column]):ax.text(v+.01,yy,f'{v:.3f}',va='center',fontsize=6)
    ax.invert_yaxis();ax.set_yticks(y,[f'{r.sampleID.replace("VISDS","")}  {r.MMR}' for r in ordered.itertuples()],fontsize=7)
    ax.set_xlim(0,1.03);ax.set_xlabel('Component value / model score',fontsize=9)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.015),fontsize=8,ncol=1)
    fig.text(.71,.978,'B  Section components and model scores',fontsize=11,weight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    ax=fig.add_axes([.74,.085,.235,.235])
    patients=pd.read_csv(CURRENT/'patient_predictions.csv')
    for label,color in [('MSS','#277F8E'),('MSI','#B65066')]:
        indices=np.flatnonzero(patients.MMR.eq(label));ax.scatter(indices,patients.iloc[indices].score,color=color,s=38,label=label)
    ax.set_xticks(range(len(patients)),patients.patientID,fontsize=8);ax.set_ylim(0,1.03);ax.set_ylabel('Mean section score');ax.set_xlabel('Patient')
    ax.set_title('C  Patient-level frozen-model scores',loc='left',fontsize=11,weight='bold',pad=12);ax.legend(frameon=False,fontsize=8)
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Figure04_paired_spatial.{ext}',dpi=230)
    plt.close(fig)
    sections.to_csv(OUT/'Figure04_section_scores.csv',index=False)
    metrics.to_csv(OUT/'Figure04_AUROC.csv',index=False)
    return vmax
