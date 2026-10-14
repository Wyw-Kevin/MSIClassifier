"""English tables and figures for the actually fitted twelve-feature model."""
from pathlib import Path
import json,hashlib,re,shutil
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils import get_column_letter
from lxml import etree as E
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import cairosvg

HERE=Path(__file__).resolve().parent
ARTICLE=HERE.parent
OLD=ARTICLE/'scRNA_AUROC_threshold_all_candidates_seed2026_20261013'
MODEL=ARTICLE/'bulk/classifier_snapshot'
SUP=HERE/'output/data';FIG=HERE/'output/figures'
SHA=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
COHORTS=['GSE39582','TCGA-COADREAD','GSE24550','GSE25071','GSE27544','GSE35896','GSE39084','GSE75316']
SCOPE={'development_OOF':'Two subtype-development cohorts (OOF)',
       'retrospective':'Six retrospective GEO cohorts','all_eight':'All eight retrospective cohorts',
       'development_apparent':'Two development cohorts (apparent)'}
COMPONENT={'Subtype':'Twelve-subtype MLP','MMR4_LOCO':'Cohort-held-out MMR4','Fusion05':'Fusion 0.50:0.50'}

def save(df,path):df.to_csv(path,index=False,float_format='%.17g')
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def export(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=260,bbox_inches='tight')
    plt.close(fig)

def figures(metrics,macro,stability,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2878A5','#B57147','#3E9188']
    retained=stability[stability.retained_bulk12].sort_values('bootstrap_inclusion_frequency',ascending=True)
    selected=metrics[metrics.split!='development_apparent']
    values=selected.pivot(index='dataset',columns='component',values='AUROC').loc[COHORTS,['Subtype','MMR4_LOCO','Fusion05']]
    fig,axes=plt.subplots(2,2,figsize=(13.5,10.5),constrained_layout=True)
    ax=axes[0,0];bars=ax.barh(retained.index,100*retained.bootstrap_inclusion_frequency,color=colors[0])
    ax.bar_label(bars,fmt='%.2f%%',padding=3,fontsize=8);ax.set_xlim(0,114);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Bootstrap inclusion frequency (%)')
    ax.set_title('A   Twelve retained subtype frequencies',loc='left',weight='bold')
    ax=axes[0,1];im=ax.imshow(values,vmin=.5,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),['Twelve-subtype MLP','MMR4 LOCO','Fusion 0.50']);ax.tick_params(axis='x',labelsize=9)
    ax.set_yticks(range(8),[c+(' (OOF)' if c in COHORTS[:2] else '') for c in COHORTS]);ax.tick_params(axis='y',labelsize=9)
    for i in range(8):
        for j in range(3):ax.text(j,i,f'{values.iloc[i,j]:.3f}',ha='center',va='center',color='white' if values.iloc[i,j]>.83 else '#20242A',fontsize=9)
    ax.set_title('B   Per-cohort AUROC',loc='left',weight='bold');fig.colorbar(im,ax=ax,shrink=.9,label='AUROC')
    ax=axes[1,0];x=np.arange(3);width=.36
    for k,(split,label) in enumerate([('development_OOF','Subtype-development cohorts (OOF)'),('retrospective','Retrospective GEO cohorts')]):
        v=macro[macro.split==split].set_index('component').loc[['Subtype','MMR4_LOCO','Fusion05'],'AUROC']
        bars=ax.bar(x+(k-.5)*width,v,width,color=colors[k],label=label);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(x,['Twelve-subtype MLP','MMR4 LOCO','Fusion 0.50:0.50']);ax.tick_params(axis='x',labelsize=9);ax.set_ylim(0,1.22)
    ax.set_ylabel('Equal-weight cohort macro-AUROC');ax.set_yticks(np.arange(0,1.01,.2));ax.legend(loc='upper left',fontsize=8,frameon=False)
    ax.set_title('C   Fixed-half fusion and components',loc='left',weight='bold')
    ax=axes[1,1];fusion=selected[selected.component=='Fusion05'].set_index('dataset').loc[COHORTS]
    x=np.arange(8)
    ax.bar(x-.18,fusion.sensitivity,.36,color=colors[0],label='Sensitivity');ax.bar(x+.18,fusion.specificity,.36,color=colors[1],label='Specificity')
    ax.set_xticks(x,COHORTS,rotation=45,ha='right');ax.tick_params(axis='x',labelsize=8);ax.set_ylim(0,1.22)
    ax.set_ylabel('Fusion at score threshold 0.5');ax.set_yticks(np.arange(0,1.01,.2));ax.legend(loc='upper left',fontsize=9,frameon=False)
    ax.set_title('D   Fixed-threshold classification',loc='left',weight='bold')
    export(fig,'Figure4_twelve_subtype_MMR4_LOCO')
    save(retained.reset_index(),FIG/'Figure4A_values.csv');save(selected,FIG/'Figure4BD_values.csv');save(macro[macro.split.isin(['development_OOF','retrospective'])],FIG/'Figure4C_values.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    allfreq=stability.sort_values('bootstrap_inclusion_frequency',ascending=True)
    ax=axes[0,0];bars=ax.barh(allfreq.index,100*allfreq.bootstrap_inclusion_frequency,color=['#2878A5' if b else '#B9C5CE' for b in allfreq.retained_bulk12])
    ax.bar_label(bars,fmt='%.2f%%',padding=3,fontsize=7.5);ax.set_xlim(0,115);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Bootstrap inclusion frequency (%)');ax.set_title('A   All 17 fixed candidates',loc='left',weight='bold')
    overlap=pd.read_csv(HERE.parent/'bulk/stability_archive/S14_replicate_overlap.csv');sizes=pd.read_csv(HERE.parent/'bulk/stability_archive/S14_selected_count_distribution.csv')
    ax=axes[0,1];ax.bar(sizes.selected_count,sizes.frequency*100,color='#2878A5');ax.set_xlabel('Number of selected subtypes');ax.set_ylabel('Resamples (%)');ax.set_xticks(sizes.selected_count)
    ax.set_title('B   Selected subtype counts',loc='left',weight='bold')
    ax=axes[1,0];ax.hist(overlap.Jaccard_with_panel12,bins=np.linspace(0,1,21),color='#B57147',edgecolor='white')
    ax.axvline(summary['Jaccard_median'],color='#555555',ls='--');ax.set(xlabel='Jaccard overlap with the full-data panel12',ylabel='Resamples');ax.set_title('C   Panel overlap',loc='left',weight='bold')
    ax=axes[1,1];bars=ax.bar(['12 selected subtypes','Exact full-data 12-subtype set'],[100*summary['size12_frequency'],100*summary['exact_panel_recovery']],color=['#2878A5','#B57147'])
    ax.bar_label(bars,fmt='%.2f%%',padding=5,fontsize=10);ax.set_ylim(0,105);ax.set_ylabel('Resamples (%)');ax.tick_params(axis='x',labelsize=8)
    ax.set_title('D   Panel size and exact-set recovery',loc='left',weight='bold')
    export(fig,'SupplementaryFigureS2_bulk12_selection_stability')
    # Selection flow: keep the single-cell18 classifier and the separate bulk17 pool explicit.
    fig,ax=plt.subplots(figsize=(12,7.2));ax.set_xlim(0,12);ax.set_ylim(0,7.2);ax.axis('off')
    def box(x,y,w,h,t,color='#EDF3F6'):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',fc=color,ec='#52636E',lw=1))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#657780','lw':1.5})
    box(.4,5.8,11.2,1,'GSE178341 reference: 71 unmerged subtype candidates (at least 50 cells)\nFrozen cell definitions and source association directions')
    box(.4,3.7,4.4,1.4,'scRNA primary branch\nHistorical source cutoff >0.65: 24 candidates\nHistorical configuration comparison: fixed18\n18 proportions + weighted malignant MLH1')
    box(6.2,3.7,5.4,1.4,'Bulk candidate branch\nSource oriented AUROC >0.70: 17 candidates\nBoth development oriented AUROCs >0.50\n12 direction-concordant subtypes')
    arrow((3,5.8),(2.6,5.15));arrow((9,5.8),(8.9,5.15))
    box(.4,1.9,4.4,1.1,'Grouped classifier OOF + retrospective scoring\nTraining-only Z-score + shrinkage LDA\nFrozen18 model transferred to spatial proxies')
    box(6.2,1.9,5.4,1.1,'GSE39582 + TCGA-COADREAD\nArchived five-fold OOF / full-fit MLP\nTraining-only Z-score, fixed 4-unit MLP')
    arrow((2.6,3.7),(2.6,3.0));arrow((8.9,3.7),(8.9,3.0))
    box(6.2,.25,5.4,.95,'Eight-cohort LOCO MMR4: fit7, score held-out1\nFixed fusion: 0.5 subtype probability + 0.5 MMR4\nTwo development + six retrospective cohorts','#F4EEE9')
    arrow((8.9,1.9),(8.9,1.25))
    ax.text(.5,.65,'Selection stability: 2,000 development-only resamples\nSource screen and measurements fixed\nPanel adoption includes post hoc configuration comparison',fontsize=9,va='center',color='#53646D')
    export(fig,'SupplementaryFigureS1_current_selection')
    for name in ['SupplementaryFigureS1_current_selection','SupplementaryFigureS2_bulk12_selection_stability']:
        for ext in ['png','svg','pdf']:shutil.copy2(FIG/(name+'.'+ext),SUP/(name+'.'+ext))

