from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from paths import PACKAGE_ROOT as P, WORK_ROOT as O, EXPORT_ROOT as E, reference_input
E.mkdir(parents=True,exist_ok=True)
mp=pd.read_csv(reference_input('mapping.csv'),index_col=0);whole=[];parents=[];within=[];summary=[];verification=[]
for d in pd.read_csv(O/'input_audit.csv').dataset:
 c=O/'cohorts'/d
 if not (c/'complete.json').exists():continue
 x=pd.read_csv(c/'final_fractions.csv',index_col=0);assert set(x)==set(mp.index)
 ids=pd.read_pickle(O/'inputs'/f'{d}_sampling.pkl').index;assert x.index.tolist()==ids.tolist()
 assert np.isfinite(x).all().all() and (x>=0).all().all() and np.allclose(x.sum(axis=1),1)
 major=pd.DataFrame({p:x[mp.index[mp.parent==p]].sum(axis=1) for p in mp.parent.unique()})
 sub=pd.DataFrame({f:x[f]/major[mp.loc[f,'parent']].replace(0,np.nan) for f in x if f!='Epi'})
 assert np.array_equal(major.Epi.to_numpy(),x.Epi.to_numpy())
 for frame,coll in [(x,whole),(major,parents),(sub,within)]:
  frame=frame.copy();frame.insert(0,'dataset',d);frame.index.name='sampleID';coll.append(frame.reset_index())
 summary.append(dict(dataset=d,n=len(x),Epi_mean_percent=x.Epi.mean()*100,Epi_min_percent=x.Epi.min()*100,Epi_max_percent=x.Epi.max()*100,Epi_SD_pp=x.Epi.std()*100,median_subtype_CV_percent=(sub.std()/sub.mean()).median()*100))
 verification.append(json.loads((c/'complete.json').read_text()))
for coll,name in [(whole,'all72_reference_relative.csv'),(parents,'major_reference_relative.csv'),(within,'subtype_over_major.csv')]:
 if coll:pd.concat(coll,ignore_index=True).to_csv(E/name,index=False)
pd.DataFrame(summary).to_csv(E/'cohort_summary.csv',index=False)
(O/'input_audit.csv').replace(E/'input_audit.csv') if False else None
pd.read_csv(O/'input_audit.csv').to_csv(E/'input_audit.csv',index=False)
(E/'verification.json').write_text(json.dumps({'completed_cohorts':len(verification),'completed_samples':sum(x['n'] for x in verification),'expected_cohorts':len(pd.read_csv(O/'input_audit.csv')),'expected_samples':int(pd.read_csv(O/'input_audit.csv')['n'].sum()),'all_complete':len(verification)==len(pd.read_csv(O/'input_audit.csv')),'details':verification},indent=2))
(E/'说明.md').write_text('''# Python BayesPrism独立队列重分析（路径适配版）

GSE18088排除。本次选定队列各自独立运行，实际完成数见verification.json。使用固定GSE178341原始counts参考：71非上皮亚型+1个Epi大类，参考亚型至少50细胞。Epi不等同于候选恶性上皮，key=None采用所有类型同等的参考更新方式，不是肿瘤专属更新。

Python包pybayesprism0.1.0独立移植，含经测试记录的本地修复及多项式抽样向量化；并非未修改官方R实现。原始源码及修改散列位于计算目录code/vendor_original与vendor_source_hashes.json。原版有采样循环端点、基因符号分类及未匹配基因过滤、最终列标签、优化目标汇总等问题。

输入按平台恢复线性尺度：RMA/ABI使用2**x，TCGA使用2**x−1，MAS5保留原线性值；不是DECODE的log/max输入。处理对象为历史基因级汇总矩阵，并非探针级重新处理，GSE39582保留原作者ComBat。连续线性输入保存；采样需整数，显式四舍五入并记录误差。非有限基因剔除记录，不补零。芯片线性强度不是RNA-seq读数；芯片运行是探索性应用，后验CV也不能解释为生物学测量误差。

默认链1000、burn-in500、thin2、alpha1、MAP sigma2；随机根种子2026，PCG64为各样本分配不重叠的2**64状态区段，初始与最终链使用持续推进的状态分配器，不使用样本/折号种子偏移。所有样本状态保存。

输出为参考覆盖范围的相对组成；大类由同一亚型预测求和，上皮在大小类中一致。subtype_over_major使用clTopLevel父类，不是中层分类。执行检查通过不等于真实细胞比例已获验证。历史反卷积、分类器和论文不覆盖。
''')
print('Postprocess',len(verification),'cohorts',flush=True)
