"""Plot the current seventeen-subtype frozen spatial transfer from audited scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cairosvg
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
sections=pd.read_csv(HERE/'spatial/section_predictions.csv').sort_values('sampleID')
patients=pd.read_csv(HERE/'spatial/patient_predictions.csv')
spots=pd.read_csv(HERE/'spatial/spot_display_values.csv.gz')
vmax=float(np.quantile(spots.weighted_MLH1_display,.99))
fig=plt.figure(figsize=(16,12.225))
grid=fig.add_gridspec(7,4,left=.025,right=.55,bottom=.05,top=.94,wspace=.18,hspace=.33)
for i,row in enumerate(sections.itertuples()):
    s=spots[spots.sampleID.eq(row.sampleID)]
    m=s.epithelial_enriched_display.astype(bool).to_numpy()
    for j in range(2):
        ax=fig.add_subplot(grid[i//2,(i%2)*2+j]);ax.scatter(s.x,s.y,s=2,c='#E9EDF1',linewidths=0)
        use=m if j==0 else ~m
        vals=s.weighted_MLH1_display if j==0 else s.local_ecology_score
        scatter=ax.scatter(s.x[use],s.y[use],c=vals[use],s=3,linewidths=0,cmap='Reds' if j==0 else 'RdYlBu_r',vmin=0,vmax=vmax if j==0 else 1)
        if j==0:mlh1_mappable=scatter
        else:ecology_mappable=scatter
        ax.set_aspect('equal');ax.invert_yaxis();ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color('#D6DDE3');spine.set_linewidth(.45)
        ax.set_title(f'{row.sampleID} | {row.MMR}\n'+('Epithelial-enriched MLH1' if j==0 else '17-subtype score'),fontsize=6.6,pad=3)
fig.text(.025,.978,'A  Spatial MLH1 expression and 17-subtype patterns',fontsize=12,weight='bold')
for mapper,y,label in [(mlh1_mappable,.56,'Weighted reconstructed MLH1'),(ecology_mappable,.11,'Local 17-subtype score')]:
    cb=fig.colorbar(mapper,cax=fig.add_axes([.572,y,.010,.31]));cb.set_label(label,fontsize=8);cb.ax.tick_params(labelsize=7);cb.solids.set_rasterized(False)
ax=fig.add_axes([.71,.41,.265,.47]);ordered=sections.sort_values('score',ascending=False);ys=np.arange(len(ordered))
for offset,column,color,label in [(-.23,'weighted_MLH1_proxy','#CA6575','Weighted MLH1'),(0,'ecology_only_score','#4F95BE','17-subtype ecology'),(.23,'score','#43A394','Combined model')]:
    ax.barh(ys+offset,ordered[column],height=.21,color=color,label=label)
    for yy,v in zip(ys+offset,ordered[column]):ax.text(v+.012,yy,f'{v:.3f}',va='center',fontsize=6)
ax.invert_yaxis();ax.set_yticks(ys,[f'{r.sampleID.replace("VISDS","")}  {r.MMR}' for r in ordered.itertuples()],fontsize=7)
ax.set_xlim(0,1.03);ax.set_xlabel('Component value / model score',fontsize=9)
ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.015),fontsize=8)
fig.text(.71,.978,'B  Section components and model scores',fontsize=11,weight='bold');ax.spines[['top','right']].set_visible(False)
ax=fig.add_axes([.74,.085,.235,.235])
for label,col in [('MSS','#277F8E'),('MSI','#B65066')]:
    idx=np.flatnonzero(patients.MMR.eq(label));ax.scatter(idx,patients.iloc[idx].score,color=col,s=38,label=label,zorder=3)
ax.set_xticks(range(len(patients)),patients.patientID,fontsize=8)
ax.set_ylim(0,1.03);ax.set_ylabel('Mean section score');ax.set_xlabel('Patient')
ax.set_title('C  Patient-level frozen-model scores',loc='left',fontsize=11,weight='bold',pad=12)
ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'Figure3_spatial_17.svg');fig.savefig(OUT/'Figure3_spatial_17.pdf')
cairosvg.svg2png(url=str(OUT/'Figure3_spatial_17.svg'),write_to=str(OUT/'Figure3_spatial_17.png'),output_width=2400)
plt.close(fig)
print('Spatial figure exported: 17-subtype scores; patient points; unchanged reconstruction.')
