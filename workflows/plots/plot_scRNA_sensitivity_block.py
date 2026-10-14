plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
colors = {'GSE178341':'#226696','GSE132465':'#b56327','GSE200997':'#39889c'}
labels = {'GSE178341':'GSE178341 grouped OOF (n=64)',
          'GSE132465':'GSE132465 retrospective (n=23)',
          'GSE200997':'GSE200997 retrospective (n=15)'}
for ds, color in colors.items():
    a = screen.loc[screen.dataset.eq(ds)].sort_values('screening_cutoff')
    b = cnv.loc[cnv.dataset.eq(ds)&cnv['mode'].eq('refit')].sort_values('CNV_quantile')
    axes[0].plot(a.screening_cutoff, a.AUROC, color=color, marker='o', label=labels[ds], lw=1.8)
    baseline = float(fixed18.loc[fixed18.dataset.eq(ds),'AUROC'].iloc[0])
    axes[0].scatter([.65],[baseline],color=color,marker='*',s=180,edgecolors='white',linewidths=.6,zorder=5)
    axes[1].plot(b.CNV_quantile, b.AUROC, color=color, marker='o', lw=1.8)
axes[0].axvline(.65, color='#888888', ls='--', lw=1)
axes[1].axvline(.95, color='#888888', ls='--', lw=1)
axes[0].set(xlabel='Oriented AUROC screening cutoff', ylabel='AUROC',
            xticks=[.60,.65,.70,.75], ylim=(.89,1.01))
axes[1].set(xlabel='Stromal-reference CNV quantile', xticks=[.90,.95,.99])
axes[1].set_xticklabels(['q90','q95','q99'])
axes[0].set_title('A   Subtype-screening threshold', loc='left', weight='bold', pad=30)
axes[1].set_title('B   CNV candidate definition', loc='left', weight='bold', pad=30)
axes[0].text(.5,1.035,'All passing candidates: 39 / 24 / 17 / 9 (no count cap)',
             transform=axes[0].transAxes,ha='center',fontsize=8.5)
axes[1].text(.5,1.035,'Fixed 18 subtypes; weighted MLH1 recalculated at each quantile',
             transform=axes[1].transAxes,ha='center',fontsize=8.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.5,.975), ncol=3, frameon=False, fontsize=9)
fig.subplots_adjust(left=.065,right=.98,top=.78,bottom=.25,wspace=.16)
fig.text(.065,.085,
    'Archived patient-grouped folds; training-only Z-score and shrinkage LDA. Historical candidate ranking and cohort-level CNV fixed.\n'
    'Stars: historical top 18 reference at 0.65; circles: all passing candidates. Dashed lines: 0.65/q95. GSE200997: two positive cases.\n'
    'Frozen-q95 input sensitivity is reported in Table S3.',
    fontsize=8.5)
for ext in ['png','svg','pdf']:
    filename = 'SupplementaryFigureS3_scRNA_threshold_CNV_sensitivity.'+ext
    fig.savefig(FIG/filename, dpi=260, bbox_inches='tight')
    shutil.copy2(FIG/filename, SUP/filename)
plt.close(fig)

