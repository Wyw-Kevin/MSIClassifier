"""Publication figures for the selected source-AUROC>0.70 seventeen-subtype panel."""
from pathlib import Path
from copy import deepcopy
import re,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve
from lxml import etree as E
import cairosvg

HERE=Path(__file__).resolve().parents[1];F=HERE/"figures";T=HERE/"tables"
F.mkdir(exist_ok=True)
metric=pd.read_csv(T/"metrics.csv");pred=pd.read_csv(T/"all_predictions.csv")
screen=pd.read_csv(T/"threshold_performance.csv");cnv=pd.read_csv(T/"CNV_sensitivity_metrics.csv")
DS=["GSE178341","GSE132465","GSE200997"];C=["#277F8E","#B65066","#6C8A44"]
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                     "xtick.labelsize":7,"ytick.labelsize":7,"svg.fonttype":"none","pdf.fonttype":42})
def export(fig,name,width=2200):
    fig.savefig(F/(name+".svg"));fig.savefig(F/(name+".pdf"))
    cairosvg.svg2png(url=str(F/(name+".svg")),write_to=str(F/(name+".png")),output_width=width)
    plt.close(fig)
def clean(ax):ax.spines[["top","right"]].set_visible(False)
def value(ds,split,name):
    return metric[metric.dataset.eq(ds)&metric.split.eq(split)&metric.model.eq(name)].iloc[0]
scopes=[("GSE178341","training_apparent"),("GSE178341","grouped_OOF"),("GSE132465","external"),("GSE200997","external")]

fig=plt.figure(figsize=(514.6/72,480/72))
gs=fig.add_gridspec(3,2,left=.14,right=.90,top=.94,bottom=.085,height_ratios=[.65,1.0,1.35],hspace=.9,wspace=.58)
for ax,cohorts,letter,title in [(fig.add_subplot(gs[0,:]),[DS[0]],"A","Development: patient-grouped OOF"),
                                (fig.add_subplot(gs[1,:]),DS[1:],"B","External evaluation")]:
    matrix=np.array([[screen[screen.dataset.eq(ds)&screen.screening_cutoff.eq(c)].iloc[0].AUROC for c in [.6,.65,.7,.75]] for ds in cohorts])
    im=ax.imshow(matrix,aspect="auto",cmap="YlGnBu",vmin=.9,vmax=1)
    ax.set_xticks(range(4),[">0.60\n39 subtypes",">0.65\n24 subtypes",">0.70\n17 subtypes",">0.75\n9 subtypes"])
    ax.set_yticks(range(len(cohorts)),cohorts);ax.tick_params(length=0,pad=5)
    ax.set_xlabel("Source oriented AUROC cutoff",labelpad=3)
    ax.set_title(letter+"   "+title,loc="left",weight="bold",pad=9)
    for row in range(len(cohorts)):
        for col in range(4):ax.text(col,row,f"{matrix[row,col]:.3f}",ha="center",va="center",fontsize=10,color="white" if matrix[row,col]>.96 else "#15232e")
    if letter=="A":
        ax.add_patch(Rectangle((1.5,-.5),1,1,fill=False,edgecolor="#C8392B",lw=2))
    else:
        ax.add_patch(Rectangle((1.5,-.5),1,2,fill=False,edgecolor="#C8392B",lw=1.4,linestyle="--"))
    for spine in ax.spines.values():spine.set_visible(False)
cb=fig.colorbar(im,cax=fig.add_axes([.925,.51,.012,.34]),ticks=[.90,.95,1]);cb.set_label("AUROC",fontsize=7);cb.ax.tick_params(labelsize=6)
ax=fig.add_subplot(gs[2,0])
for off,name,color,label in [(-.12,"Unweighted","#8d9298","Mean MLH1"),(.12,"Joint",C[0],"Weighted MLH1")]:
    rows=[value(ds,sp,name) for ds,sp in scopes];y=np.array([r.AUROC for r in rows]);lo=np.array([r.CI_low for r in rows]);hi=np.array([r.CI_high for r in rows])
    ax.errorbar(np.arange(4)+off,y,yerr=[y-lo,hi-y],fmt="o",ms=3,capsize=2,color=color,label=label)
ax.set_xticks(range(4),["GSE178341\nApparent","GSE178341\nOOF","GSE132465\nExternal","GSE200997\nExternal"],fontsize=5.7)
ax.set(ylabel="AUROC",ylim=(.5,1.04));ax.legend(frameon=False,fontsize=6,loc="lower left");ax.set_title("C   MLH1 weighting",loc="left",weight="bold");clean(ax)
ax=fig.add_subplot(gs[2,1])
for ds,color in zip(DS,C):
    sp="grouped_OOF" if ds==DS[0] else "external"
    z=pred[pred.dataset.eq(ds)&pred.split.eq(sp)&pred.model.eq("Joint")]
    x,y,_=roc_curve(z.y,z.score);ax.plot(x,y,color=color,lw=1,label=f"{ds}: {value(ds,sp,'Joint').AUROC:.3f}")
ax.plot([0,1],[0,1],"--",color="#aaa",lw=.7);ax.set(xlabel="False positive rate",ylabel="True positive rate",xlim=(-.02,1.02),ylim=(-.02,1.02))
ax.legend(frameon=False,fontsize=6,loc="lower right");ax.set_title("D   17-subtype joint-model ROC",loc="left",weight="bold");clean(ax)
export(fig,"Figure2_main_AUROC_cutoffs")

genes=pd.read_csv(HERE/"inputs/MMR_gene_expression.csv");q=pd.read_csv(HERE/"inputs/cnv_quantiles.csv");rng=np.random.default_rng(2026)
fig,axes=plt.subplots(3,2,figsize=(514.6/72,568/72));fig.subplots_adjust(left=.12,right=.98,top=.95,bottom=.075,hspace=.55,wspace=.37)
for ax,gene,letter in zip(axes.flat,["MLH1","MSH2","MSH6","PMS2"],"EFGH"):
    for i,ds in enumerate(DS):
        for j,label in enumerate(["MSS","MSI"]):
            vals=genes[genes.dataset.eq(ds)&genes.MMR.eq(label)&genes.gene.eq(gene)].mean_expression.to_numpy();pos=i+(j-.5)*.30;color=C[j]
            ax.boxplot([vals],positions=[pos],widths=.24,showfliers=False,patch_artist=True,
                       boxprops={"facecolor":color,"edgecolor":color,"alpha":.3},medianprops={"color":"#222"})
            ax.scatter(pos+rng.uniform(-.05,.05,len(vals)),vals,s=7,color=color,alpha=.7,linewidths=.15,edgecolors="white",label=label if i==0 else None)
    ax.set_xticks(range(3),DS,fontsize=6);ax.set_ylabel("Mean normalized expression",fontsize=7)
    ax.set_title(letter+"   "+gene,loc="left",weight="bold");clean(ax)
axes[0,0].legend(frameon=False,fontsize=6,loc="upper right")
ax=axes[2,0]
for i,ds in enumerate(DS):
    frame=pd.read_csv(T/(ds+"_primary17_features.csv"))
    for j,y in enumerate([0,1]):
        vals=frame[frame.y.eq(y)].weighted_MLH1.to_numpy();pos=i+(j-.5)*.30;color=C[j]
        ax.boxplot([vals],positions=[pos],widths=.24,showfliers=False,patch_artist=True,boxprops={"facecolor":color,"alpha":.3},medianprops={"color":"#222"})
        ax.scatter(pos+rng.uniform(-.05,.05,len(vals)),vals,s=7,color=color,alpha=.7,label=["MSS","MSI"][j] if i==0 else None)
ax.set_xticks(range(3),DS,fontsize=6);ax.set_ylabel("Weighted MLH1");ax.set_title("I   Sample-level weighted MLH1",loc="left",weight="bold");ax.legend(frameon=False,fontsize=6);clean(ax)
ax=axes[2,1];models=["Joint","Subtype","MLH1"]
for i,(ds,color) in enumerate(zip(DS,C)):
    sp="grouped_OOF" if ds==DS[0] else "external";rr=[value(ds,sp,n) for n in models]
    y=np.array([r.AUROC for r in rr]);lo=np.array([r.CI_low for r in rr]);hi=np.array([r.CI_high for r in rr])
    ax.errorbar(np.arange(3)+(i-1)*.18,y,yerr=[y-lo,hi-y],fmt="o",ms=3,capsize=2,color=color,label=ds)
ax.set_xticks(range(3),["17 subtypes\n+ weighted MLH1","17 subtypes\nonly","Weighted\nMLH1 only"],fontsize=6)
ax.set(ylabel="AUROC",ylim=(.3,1.04));ax.legend(frameon=False,fontsize=6,loc="lower left");ax.set_title("J   Matched component models",loc="left",weight="bold");clean(ax)
export(fig,"Figure2_continued_17")

# Keep the approved study-design layout; replace text only.
root=E.fromstring((HERE/"inputs/Figure1_source.svg").read_bytes())
for node in root.xpath('//*[local-name()="text"]'):
    text="".join(node.itertext())
    if text=="Retrospective evaluation":new="External evaluation"
    elif text=="Schematic glyphs; external evaluation is retrospective. Spatial groups are inferred, and measurement equivalence is unestablished.":new="scRNA: GSE178341 development; GSE132465/GSE200997 external evaluation. Spatial groups are inferred; proxy equivalence is unestablished."
    elif text=="71 → 24 → top 18 subtype proportions":new="71 → AUROC >0.70 → 17 subtypes"
    elif text=="Fixed 18-subtype panel":new="17 subtypes; cutoff selected by OOF"
    else:new=text.replace("18 subtype","17 subtype").replace("18 fractions","17 fractions").replace("18-subtype","17-subtype")
    if new!=text:
        for child in list(node):node.remove(child)
        node.text=new
        if len(new)>35:node.set("font-size","13")
blob=E.tostring(root);(F/"Figure1_study_design_17.svg").write_bytes(blob)
cairosvg.svg2png(bytestring=blob,write_to=str(F/"Figure1_study_design_17.png"),output_width=2200)
cairosvg.svg2pdf(bytestring=blob,write_to=str(F/"Figure1_study_design_17.pdf"))

# Source comparison cells are kept; update this study's two changed scRNA cells and their colors.
root=E.fromstring((HERE/"inputs/Figure5_source.svg").read_bytes());ns={"s":"http://www.w3.org/2000/svg"}
before_bulk=E.tostring(root.xpath('//*[@id="axes_3"]')[0])
changes=[]
for text in root.xpath('//s:text',namespaces=ns):
    old="".join(text.itertext())
    if old in {"0.978","0.961"}:
        target=value("GSE178341","grouped_OOF","Joint").AUROC if old=="0.978" else value("GSE132465","external","Joint").AUROC
        text.text=f"{target:.3f}";changes.append((old,text.text))
# The current-study heatmap uses rects; only its first two cells change.
rects=root.xpath('//*[@id="axes_2_new_axes_1"]//s:rect',namespaces=ns)
assert len(rects)>=3
for index,target in [(0,value("GSE178341","grouped_OOF","Joint").AUROC),(1,value("GSE132465","external","Joint").AUROC)]:
    assert float(rects[index].get("y"))<30
    rects[index].set("fill",matplotlib.colors.to_hex(plt.get_cmap("YlGnBu")((target-.65)/.35)))
assert len(changes)==2
assert before_bulk==E.tostring(root.xpath('//*[@id="axes_3"]')[0])
blob=E.tostring(root);(F/"Figure5_comparison_17.svg").write_bytes(blob)
cairosvg.svg2png(bytestring=blob,write_to=str(F/"Figure5_comparison_17.png"),output_width=1834,output_height=1712)
cairosvg.svg2pdf(bytestring=blob,write_to=str(F/"Figure5_comparison_17.pdf"))

fig,axes=plt.subplots(1,2,figsize=(11.2,4));fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.86,wspace=.28)
for ds,color in zip(DS,C):
    z=screen[screen.dataset.eq(ds)].sort_values("screening_cutoff")
    axes[0].plot(z.screening_cutoff,z.AUROC,"o-",color=color,label=ds+" OOF" if ds==DS[0] else ds+" external")
    for mode,style in [("refit","o-"),("frozen_q95","x--")]:
        z=cnv[cnv.dataset.eq(ds)&cnv.split.ne("training_apparent")&cnv["mode"].eq(mode)].sort_values("quantile")
        axes[1].plot(z["quantile"],z.AUROC,style,color=color,alpha=1 if mode=="refit" else .65)
axes[0].set(xticks=[.60,.65,.70,.75],xlabel="Source oriented AUROC cutoff",ylabel="AUROC",ylim=(.89,1.01))
axes[0].axvline(.7,color="#aaa",ls=":",lw=.8);axes[0].set_title("A   All candidates above each cutoff",loc="left",weight="bold")
axes[1].set(xticks=[.9,.95,.99],xticklabels=["q90","q95","q99"],xlabel="Stromal-reference CNV quantile",ylabel="AUROC",ylim=(.89,1.01))
axes[1].set_title("B   Current 17-subtype CNV sensitivity",loc="left",weight="bold")
fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.02),ncol=3,frameon=False,fontsize=8)
fig.text(.5,.145,"39 / 24 / 17 / 9 subtypes at the four cutoffs; 17 selected using development OOF. CNV: circles refit; crosses frozen q95.",ha="center",fontsize=7)
for ax in axes:clean(ax)
export(fig,"SupplementaryFigureS3_17",width=2400)

# Selection diagram for the separate supplement; all figure labels are English.
fig,ax=plt.subplots(figsize=(10.5,4.2));ax.set_axis_off()
boxes=[(.02,.61,.23,.25,"GSE178341 reference\n71 unmerged candidate subtypes\nAt least 50 reference cells"),
       (.38,.61,.24,.25,"Source direction + AUROC\nCutoffs: >0.60 / >0.65 / >0.70 / >0.75\n39 / 24 / 17 / 9 passing subtypes"),
       (.76,.61,.22,.25,"Development grouped OOF\nFixed shrinkage LDA + weighted MLH1\nHighest AUROC: >0.70 (0.9803)"),
       (.23,.12,.26,.25,"Single-cell primary model\nAll 17 subtypes with AUROC >0.70\n+ weighted MLH1"),
       (.61,.12,.28,.25,"Bulk direction concordance\nAUROC >0.50 in both development cohorts\n12 subtypes + fixed0.5 MMR4 fusion")]
for x,y,w,h,label in boxes:
    ax.add_patch(Rectangle((x,y),w,h,facecolor="#EDF4F7",edgecolor="#6b8190",lw=.8))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8)
for a,b in [((.25,.735),(.38,.735)),((.62,.735),(.76,.735)),((.87,.61),(.36,.37)),((.87,.61),(.75,.37))]:
    ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#5b6e79","lw":1})
fig.tight_layout();export(fig,"SupplementaryFigureS1_17",width=2200)
print("Figures2,1,5,S1,S3 exported from current17 numerical inputs.")
