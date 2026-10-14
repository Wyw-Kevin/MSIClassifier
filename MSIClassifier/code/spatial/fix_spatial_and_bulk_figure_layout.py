import os
"""Restore paired spatial displays with current data; repair Figure 5A SVG layout."""
from pathlib import Path
from zipfile import ZipFile
import json
import shutil
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from lxml import etree as E
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import subprocess

FINAL=Path(os.environ.get('MSI_PROJECT_ROOT', str(Path(__file__).resolve().parent/'resources/project_snapshot'))).resolve()/'最终模型'
ARTICLE=FINAL/'report/article'
DOC=ARTICLE/'MMR多转录组分类器_第二版.docx'
OUT=ARTICLE/'figure_layout_corrections'
SOURCE=FINAL/'data/stRNA/result/weighted_mlh1_transfer_seed2026'
CURRENT=ARTICLE/'weighted_spatial_update'
BACKUP=ARTICLE/'backups/MMR多转录组分类器_第二版_before_spatial_layout_restore.docx'
NS={'s':'http://www.w3.org/2000/svg','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

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
        s['local_ecology_score']=model['pipeline'].predict_proba(x[model['features']])[:,1]
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
            label='Epithelial-enriched MLH1' if j==0 else '18-subtype score'
            ax.set_title(f'{row.sampleID} | {row.MMR}\n{label}',fontsize=6.6,pad=3)
    fig.text(.025,.978,'A  Spatial MLH1 expression and 18-subtype patterns',fontsize=12,weight='bold')
    for mapper,y,label in [(mlh1_mappable,.56,'Weighted reconstructed MLH1'),(ecology_mappable,.11,'Local 18-subtype score')]:
        cb=fig.colorbar(mapper,cax=fig.add_axes([.572,y,.010,.31]));cb.set_label(label,fontsize=8);cb.ax.tick_params(labelsize=7);cb.solids.set_rasterized(False)
    ax=fig.add_axes([.71,.41,.265,.47]);ordered=sections.sort_values('score',ascending=False);y=np.arange(len(ordered))
    for offset,column,color,label in [(-.23,'weighted_MLH1_proxy','#CA6575','Weighted MLH1'),(0,'ecology_only_score','#4F95BE','18-subtype ecology'),(.23,'score','#43A394','Combined model')]:
        ax.barh(y+offset,ordered[column],height=.21,color=color,label=label)
        for yy,v in zip(y+offset,ordered[column]):ax.text(v+.01,yy,f'{v:.3f}',va='center',fontsize=6)
    ax.invert_yaxis();ax.set_yticks(y,[f'{r.sampleID.replace("VISDS","")}  {r.MMR}' for r in ordered.itertuples()],fontsize=7)
    ax.set_xlim(0,.85);ax.set_xlabel('Component value / model score',fontsize=9)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.015),fontsize=8,ncol=1)
    fig.text(.71,.978,'B  Section components and model scores',fontsize=11,weight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    ax=fig.add_axes([.74,.085,.235,.235])
    values=[float(metrics.loc[metrics.level.eq(level)&metrics.model.eq('Weighted reconstructed transfer'),'AUROC'].iloc[0]) for level in ['Section','Patient']]
    ax.bar([0,1],values,color=['#299C8E','#E67B60'],width=.56)
    for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=10,weight='bold')
    ax.set_xticks([0,1],['Sections\n(n=14)','Patients\n(n=7)'],fontsize=9);ax.set_ylim(0,1.14);ax.set_ylabel('Descriptive AUROC')
    ax.set_title('C  Exploratory spatial transfer',loc='left',fontsize=11,weight='bold',pad=12)
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Figure04_paired_spatial.{ext}',dpi=230)
    plt.close(fig)
    sections.to_csv(OUT/'Figure04_section_scores.csv',index=False)
    metrics.to_csv(OUT/'Figure04_AUROC.csv',index=False)
    return vmax

def bulk():
    source=OUT/'Figure05_before.svg'
    if not source.exists():
        with ZipFile(DOC) as z:source.write_bytes(z.read('word/media/image10.svg'))
    root=E.fromstring(source.read_bytes());ax=root.xpath('//*[@id="axes_1"]')[0]
    legend=ax.xpath('.//*[@id="legend_1"]')[0]
    assert legend.get('transform') is None
    legend.set('transform','translate(489 -85)')
    removed=[]
    for p in list(ax.xpath('.//s:g[starts-with(@id,"ytick_")]//s:path[@clip-path]',namespaces=NS)):
        assert 'stroke: #d9dee3' in p.get('style','')
        removed.append(p.get('d'));p.getparent().remove(p)
    assert len(removed)==6
    # Numerical labels and all non-A panels must remain byte-identical as XML trees.
    old=E.fromstring(source.read_bytes())
    assert root.xpath('//s:text/text()',namespaces=NS)==old.xpath('//s:text/text()',namespaces=NS)
    for g in old.xpath('//s:g[starts-with(@id,"axes_")]',namespaces=NS):
        if g.get('id')!='axes_1':assert E.tostring(g)==E.tostring(root.xpath('//*[@id=$id]',id=g.get('id'))[0])
    data=E.tostring(root)
    (OUT/'Figure05_legend_fixed.svg').write_bytes(data)
    cairosvg.svg2pdf(bytestring=data,write_to=str(OUT/'Figure05_legend_fixed.pdf'))
    # Adjacent colourbar tiles need crisp raster edges to avoid Cairo antialias seams.
    preview=E.fromstring(data)
    preview.xpath('//*[@id="QuadMesh_2"]')[0].set('shape-rendering','crispEdges')
    cairosvg.svg2png(bytestring=E.tostring(preview),write_to=str(OUT/'Figure05_legend_fixed.png'),scale=2.5)
    panel=E.fromstring(data);panel.set('viewBox','0 0 773.46852 236');panel.set('height','236pt')
    (OUT/'Figure05A_legend_fixed.svg').write_bytes(E.tostring(panel))
    cairosvg.svg2pdf(bytestring=E.tostring(panel),write_to=str(OUT/'Figure05A_legend_fixed.pdf'))
    cairosvg.svg2png(bytestring=E.tostring(panel),write_to=str(OUT/'Figure05A_legend_fixed.png'),scale=2.5)
    return len(removed)

def publish(vmax,removed):
    assert not BACKUP.exists(),'Inspect existing publication before rerunning'
    shutil.copy2(DOC,BACKUP);st=DOC.stat()
    with ZipFile(DOC) as z:infos=z.infolist();blobs={i.filename:z.read(i.filename) for i in infos}
    root=E.fromstring(blobs['word/document.xml']);old=E.fromstring(blobs['word/document.xml']);ps=root.xpath('//w:body/w:p',namespaces=NS)
    def setp(i,t):
        nodes=ps[i].xpath('.//w:t',namespaces=NS);assert nodes;nodes[0].text=t
        for n in nodes[1:]:n.text=''
    txt=''.join(ps[64].xpath('.//w:t/text()',namespaces=NS))
    txt=txt.replace('图4A显示各切片的加权重建MLH1空间分布，图4B展示冻结模型与仅18亚型模型的切片连续分数。',
        '图4A沿用成对切片点图，分别显示上皮富集spot的加权重建MLH1及其余spot的18亚型局部评分；该分区仅用于显示，不改变切片特征或预测。图4B展示切片级加权MLH1代理值、18亚型模型及联合模型分数，图4C展示联合模型的切片级与患者级AUROC。')
    txt=txt.replace('（图4C）。空间加权MLH1范围','。空间加权MLH1范围')
    setp(64,txt)
    setp(67,'图4. 18亚型与恶性上皮加权MLH1模型的空间探索性迁移。（A）每张切片并排展示两个点图：左图在上皮丰度占全部细胞丰度比例不低于该切片第75百分位的spot上显示加权重建MLH1，右图在其余spot显示18亚型模型的局部评分；未着色spot以浅灰色表示。上皮富集规则仅用于显示，不用于筛选切片模型的输入。MLH1采用共同色标，上限为全部spot显示值的第99百分位，超出部分仅在显示时截断；18亚型评分色标为0–1。局部评分仅为特征定位示意，未经spot级MSI验证。（B）当前重算结果的切片级MLH1代理值、仅18亚型及联合模型分数；MLH1代理表达值与模型分数具有不同含义，不用于直接比较数值大小。（C）联合模型的切片级和患者级描述性AUROC；患者分数为两张切片预测概率的均值。共14张切片、7位患者，其中2位为公开分组推断的MSI，结果不作为独立临床验证。')
    for tag in ['pPr','rPr','tbl','drawing','sectPr']:
        assert [E.tostring(n) for n in old.xpath('//w:'+tag,namespaces=NS)]==[E.tostring(n) for n in root.xpath('//w:'+tag,namespaces=NS)]
    blobs['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    for media,filename in [('image7.png','Figure04_paired_spatial.png'),('image8.svg','Figure04_paired_spatial.svg'),('image9.png','Figure05_legend_fixed.png'),('image10.svg','Figure05_legend_fixed.svg')]:
        blobs['word/media/'+media]=(OUT/filename).read_bytes()
    tmp=DOC.with_suffix('.layout.tmp')
    with ZipFile(tmp,'w') as z:
        for info in infos:z.writestr(info,blobs[info.filename])
    with ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(DOC);os.utime(DOC,ns=(st.st_atime_ns,st.st_mtime_ns))
    (OUT/'verification.json').write_text(json.dumps(dict(document=str(DOC),backup=str(BACKUP),
        spatial_source=str(CURRENT),spatial_scores_unchanged=True,spatial_display_partition_only=True,
        MLH1_display_vmax=vmax,bulk_A_grid_lines_removed=removed,bulk_other_panels_unchanged=True,
        formatting_and_metadata_preserved=True),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    vmax=spatial();removed=bulk()
    if args.publish:publish(vmax,removed)
    print(OUT)

if __name__=='__main__':main()
