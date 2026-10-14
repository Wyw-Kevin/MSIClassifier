"""Sentence-red revision of a new DOCX copy; preserve unrelated content/styles."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import difflib,json,re,hashlib
import pandas as pd
from lxml import etree as E
from PIL import Image

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source_archive/source_manuscript.docx'
DEST=HERE/'MMR分模态分类器_第三版_十二亚型与留一队列MMR4_句子标红_参考文献颜色修正.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q=lambda name:'{'+NS['w']+'}'+name
txt=lambda node:''.join(node.xpath('.//w:t/text()',namespaces=NS))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with ZipFile(SOURCE) as z:infos=z.infolist();original={i.filename:z.read(i.filename) for i in infos}
root=E.fromstring(original['word/document.xml']);body=root.find('w:body',NS);blocks=list(body)
oldtexts={i:txt(b) for i,b in enumerate(blocks)}
summary=json.loads((HERE/'stability/summary.json').read_text())
freqmin=100*summary['inclusion_min'];freqmax=100*summary['inclusion_max']
changes={}

def replace(index,old,new):
    text=changes.get(index,oldtexts[index]);assert text.count(old)==1,(index,old)
    changes[index]=text.replace(old,new)

def assign(index,new):changes[index]=new

# Only bulk sentences change in the abstract and shared context.
replace(5,'bulk分支以两个开发队列中与单细胞方向一致、并且定向AUROC>0.60的10个亚型构建模型，与留一队列拟合的MMR4评分按0.5∶0.5固定融合。',
        'bulk分支以两个开发队列中与单细胞关联方向一致的12个亚型构建模型，与留一队列拟合的MMR4评分按0.5∶0.5固定融合。')
replace(6,'bulk十亚型MLP与留一队列MMR4的固定等权融合，在两个亚型开发队列及六个GEO队列中的队列等权宏平均AUROC分别为0.940和0.937。',
        'bulk十二亚型MLP与留一队列MMR4的固定等权融合，在两个开发队列的五折折外评价及六个GEO队列的回顾性评价中，队列等权宏平均AUROC分别为0.934和0.939。')
replace(6,'该分析在既定亚型面板和分类器预测条件下逐队列重新拟合MMR4阈值。',
        'MMR4阈值在每轮的其余七个队列中拟合，并用于被留出队列的融合评分。')
replace(6,'开发队列内2,000次bootstrap中，十亚型的入选频率为53.95%–100.00%。',
        f'开发队列内2,000次bootstrap中，十二亚型的入选频率为{freqmin:.2f}%–{freqmax:.2f}%。')
replace(15,'bulk RNA分支联合10个同向亚型与MMR4信号','bulk RNA分支联合12个同向亚型与MMR4信号')
replace(19,'bulk分支在两个开发队列内筛选10个同向亚型','bulk分支在两个开发队列内筛选12个同向亚型')
replace(20,'十亚型面板','十二亚型面板')
replace(20,'主要融合采用既定MLP配置及固定0.5∶0.5比例，其余10种分类器用于敏感性分析。',
        '主要融合采用既定MLP配置及固定0.5∶0.5比例，亚型分类器在两个开发队列的训练折及完整开发集中分别重新拟合。')
replace(22,'（B）bulk在两个开发队列内筛选10个同向亚型并拟合分类器','（B）bulk从单细胞定向AUROC>0.70的17个候选中筛选12个同向亚型并拟合分类器')
replace(22,'开发数据内的十亚型入选频率','开发数据内的十二亚型入选频率')
replace(26,'两个开发队列用于十亚型筛选及亚型分类器拟合','两个开发队列用于十二亚型筛选及亚型分类器拟合')

assign(63,'bulk亚型候选来自GSE178341中定向AUROC>0.70的17个非上皮亚型，关联方向按单细胞开发队列固定。'
    '分别计算TCGA-COADREAD和GSE39582的单亚型定向AUROC，保留两个队列中与单细胞关联方向均一致的亚型，形成12亚型面板。'
    '方向一致性以两开发队列定向AUROC均严格>0.50为判据。'
    '分类器沿用存档的五折样本划分，特征Z-score参数仅由合并的开发训练部分估计；GEO队列使用完整开发集亚型模型及其标准化参数。'
    '主要MLP采用单隐层4个神经元、ReLU激活、L2正则化参数alpha=1、Adam优化及最大5,000次迭代；随机种子为2026。'
    '十二亚型模型在每个训练折及完整开发集中分别拟合，并使用原生预测概率。'
    '十二亚型面板与既有反卷积特征固定，五折评价反映该特征表示下的下游分类器性能[30]。')
replace(64,'固定24个候选及GSE178341方向，每次重新计算两开发队列的定向AUROC，并按均>0.60的规则确定入选集合。',
        '固定17个候选及GSE178341方向，每次重新计算两开发队列的定向AUROC，并按均>0.50的方向一致性规则确定入选集合。')
replace(65,'P为十亚型分类器的预测概率','P为十二亚型分类器的预测概率')
replace(66,'留一队列评价保留十亚型面板、反卷积特征及亚型模型预测','留一队列评价保留十二亚型面板、反卷积特征及亚型模型预测')
replace(66,'入选10个亚型的频率定义为每次bootstrap恰好筛选出10个亚型的比例，集合重合度采用Jaccard指数。',
        '入选数量的抽样分布描述面板规模的稳定性，集合重合度采用Jaccard指数。')
replace(66,'原十亚型集合的完全复现要求全部原亚型入选且其他候选均不入选，相关结果保留于补充表S14。',
        '完整十二亚型集合的复现要求全部面板亚型入选且其他候选均不入选，入选12个亚型的数量频率与完整集合复现率分别列于补充表S14。')
replace(66,'逐队列报告AUROC、固定0.5判定界值下的平衡准确率（BA）、敏感度、特异度及Brier分数，并按队列等权汇总。',
        '逐队列报告AUROC、固定0.5判定界值下的平衡准确率（BA）、敏感度、特异度及Brier分数，并按队列等权汇总。'
        'AUROC区间与7亚型和12亚型的配对差值采用2,000次队列内结局分层样本bootstrap估计，固定预测分数及模型，不纳入重新拟合或特征筛选的不确定性。')
replace(71,'bulk十亚型MLP','bulk十二亚型MLP')
replace(74,'十亚型MLP','十二亚型MLP')
replace(78,'十亚型分类器','十二亚型分类器')
replace(78,'补充表S13中的分类器范围为11种固定参数分类器的描述性取值范围，不表示置信区间。',
        '补充表S13列示十二亚型MLP的逐队列结果、宏平均、逐样本评分及7亚型和12亚型的配对比较；历史十亚型与11种分类器结果单独存档。')

panel=json.loads((HERE/'model_computation/panels_locked_before_fit.json').read_text())['panel12']
assign(111,'在单细胞定向AUROC>0.70的17个候选中，12个亚型在TCGA-COADREAD与GSE39582均保持GSE178341的关联方向，'
    '分别为'+ '、'.join(panel)+'（表4）。其中cTNI16与阳性相关，其余11个与阴性相关。'
    '两开发队列的单亚型定向AUROC均>0.50，逐队列结果见补充表S11。')
assign(112,f'两个开发队列内2,000次分层bootstrap中，十二亚型的入选频率为{freqmin:.2f}%–{freqmax:.2f}%（图4A、表4及补充表S14）。'
    'cS27和cM06的入选频率均为100.00%，cS17和cM05分别为89.05%和92.80%，显示面板内不同亚型对抽样变化的响应存在差异。'
    '入选亚型数的中位数为12，第2.5至97.5百分位为11–13；全部17个候选的入选频率、集合重合度及完整十二亚型集合的复现情况见补充图S2与补充表S14。'
    '该分析固定单细胞候选筛选、方向及既有反卷积丰度，描述这一条件下的开发集筛选稳定性。')
assign(113,'十二亚型MLP在两个开发队列五折折外评价中的队列等权宏平均AUROC为0.8095，在六个GEO队列的回顾性评价中为0.7690（表2B、图4B–C）。'
    'GSE39582与TCGA-COADREAD的亚型模型折外AUROC分别为0.8791和0.7399；逐队列指标与逐样本评分见补充表S13。'
    '完整开发集拟合后的训练样本宏平均AUROC为0.8414，作为拟合结果单独列示。')
replace(114,'bulk十亚型','bulk十二亚型')
assign(116,'注：各亚型的关联方向以GSE178341为参考固定。bulk候选为单细胞定向AUROC>0.70的17个亚型，'
    '保留TCGA-COADREAD与GSE39582中定向AUROC均严格>0.50的12个亚型，该判据表示关联方向一致。'
    'bootstrap入选频率来自两个开发队列内按队列和结局分层的2,000次重抽样，候选及参考方向保持固定。'
    '六个回顾性GEO队列的结局标签不参与亚型筛选。全部候选的入选频率、入选数量及集合重合度见补充表S14。')
assign(119,'十二亚型MLP与留一队列MMR4按0.5∶0.5固定融合后，两个开发队列及六个GEO队列的等权宏平均AUROC分别为0.9335和0.9387（表2B、图4B–C）。'
    '相较MMR4单分支，宏平均AUROC点估计分别增加0.0315和0.0335。'
    'GSE39582与TCGA-COADREAD的融合AUROC分别为0.9374和0.9296；六个GEO队列的融合AUROC范围为0.7821–0.9943。'
    '上述结果来自固定亚型面板及分类器预测条件下的逐队列MMR4重估，队列分组沿用亚型模型的开发用途。')
assign(120,'固定0.5判定界值下，等权融合在两个开发队列及六个GEO队列的宏平均平衡准确率分别为0.8555和0.8115，'
    '宏平均Brier分数分别为0.0705和0.0873（表2B）。逐队列敏感度与特异度见图4D。'
    '7亚型与12亚型固定融合的配对AUROC差值区间在开发集及回顾性队列中均包含0，配置间的判别差异具有不确定性（补充表S13）。')
assign(122,'图4. 十二亚型模型、留一队列MMR4及固定等权融合。'
    '（A）两个开发队列内2,000次分层bootstrap所得十二亚型入选频率，候选及方向固定。'
    '（B）十二亚型MLP、MMR4及0.5∶0.5融合在八个队列中的AUROC；开发队列使用五折折外概率，GEO队列使用完整开发集亚型模型概率，MMR4均采用相应留一队列评分。'
    '（C）三种评分在两个开发队列及六个GEO队列的等权宏平均AUROC。'
    '（D）固定0.5判定界值下，等权融合的逐队列敏感度与特异度。'
    'MMR4留一队列分析保持亚型面板、反卷积测量及亚型模型预测固定；全部候选频率、入选数量及完整集合复现率见补充图S2与补充表S14。')
replace(126,'十亚型MLP','十二亚型MLP')
replace(132,'十亚型MLP','十二亚型MLP')
replace(135,'bulk分支由两个开发队列筛选10个同向亚型','bulk分支由两个开发队列筛选12个同向亚型')
replace(135,'亚型入选频率及筛选阈值敏感性显示','亚型入选频率及集合重合度显示')
replace(138,'0.65和0.60为探索性筛选标准。',
        '单细胞主面板及bulk候选的确定均包含探索性筛选，bulk十二亚型配置的确定包含事后配置比较。')
replace(138,'开发数据内的bootstrap揭示了bulk亚型入选频率的异质性，部分候选对抽样及阈值变化较敏感；反卷积测量误差与完整筛选流程的不确定性仍需进一步评价。',
        '开发数据内的bootstrap评价固定候选和丰度下的入选频率，未纳入单细胞候选发现、反卷积误差及完整开发流程的不确定性。')
replace(140,'十亚型面板','十二亚型面板')
replace(144,'S12–S13记录MMR4留一队列阈值及11种分类器的逐队列结果。',
        'S12记录MMR4留一队列阈值，S13记录十二亚型MLP、MMR4和融合的逐队列结果、评分及7亚型和12亚型的配对比较。')
replace(144,'S14列示全部候选的2,000次bootstrap入选频率、入选数量、阈值敏感性、集合重合度及原十亚型集合的完全复现情况。',
        'S14列示17个候选的2,000次bootstrap入选频率、入选数量、集合重合度及完整十二亚型集合的复现情况，历史十亚型与11种分类器记录另行存档。')
assign(145,'补充图S1. 细胞亚型候选筛选与评价流程。'
    'scRNA主分支由71个候选形成24个候选及固定18亚型面板；bulk分支以单细胞定向AUROC>0.70的17个亚型为候选，'
    '保留两个开发队列中关联方向一致的12个亚型，两队列定向AUROC均严格>0.50。'
    '亚型分类器在两个开发队列的训练折及完整开发集中分别拟合，MMR4在八轮留一队列分析中分别用其余七个队列拟合阈值，两分支按0.5∶0.5固定融合。'
    '单亚型入选频率见图4A，完整集合重合度见补充表S14及图S2。')
assign(147,'补充图S2. bulk十二亚型筛选的抽样敏感性。'
    '固定17个候选及GSE178341方向，在两个开发队列内按队列和结局分层开展2,000次bootstrap，每次重新应用两队列定向AUROC均>0.50的方向一致性规则。'
    '（A）全部17个候选的入选频率；（B）每次入选亚型数；（C）与完整开发集十二亚型集合的Jaccard重合度；（D）入选12个亚型的数量频率及完整十二亚型集合复现率。'
    '入选数量中位数为12，第2.5至97.5百分位为11–13，Jaccard中位数为1.000。'
    '入选12个亚型的数量频率为77.90%（1,558/2,000），完整十二亚型集合的复现率为77.20%（1,544/2,000）。'
    '两项指标分别描述面板规模与具体组成的抽样稳定性，结果以固定候选方向及反卷积丰度为条件。')

audit=[];red_ranges={};equal_ranges={}

def style(run):
    rp=run.find('w:rPr',NS)
    return E.tostring(rp,method='c14n') if rp is not None else None

def new_run(p,value,template,red=False):
    run=E.SubElement(p,q('r'));rp=deepcopy(template) if template is not None else E.Element(q('rPr'))
    if red:
        for color in rp.findall('w:color',NS):rp.remove(color)
        E.SubElement(rp,q('color'),{q('val'):'FF0000'})
    run.append(rp);t=E.SubElement(run,q('t'));t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=value

def revise(p,new,index):
    old=txt(p);assert old!=new
    runs=[];offset=0
    for r in p.findall('w:r',NS):
        ts=r.findall('w:t',NS);assert len(ts)==1 and set(E.QName(c).localname for c in r)<={'rPr','t'}
        length=len(ts[0].text or '');runs.append((offset,offset+length,r));offset+=length
    assert offset==len(old)
    sentences=lambda t:re.findall(r'[^。！？]+[。！？]?|[。！？]',t)
    a=sentences(old);b=sentences(new)
    apos=[0]
    for s in a:apos.append(apos[-1]+len(s))
    for r in p.findall('w:r',NS):p.remove(r)
    position=0;unchanged=[];changed=[]
    for operation,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if operation=='equal':
            start,end=apos[i],apos[j];newstart=position
            for left,right,r in runs:
                lo,hi=max(start,left),min(end,right)
                if lo<hi:
                    clone=deepcopy(r);t=clone.find('w:t',NS);t.text=old[lo:hi];t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');p.append(clone)
            position+=end-start;unchanged.append([start,end,newstart,position])
        else:
            start=apos[i];template=next((r.find('w:rPr',NS) for left,right,r in runs if left<=min(start,len(old)-1)<right),runs[0][2].find('w:rPr',NS))
            for value in b[k:l]:
                new_run(p,value,template,True);changed.append([position,position+len(value)]);position+=len(value)
            audit.append({'block':index,'before':''.join(a[i:j]),'after':''.join(b[k:l]),'unit':'changed sentence'})
    assert txt(p)==new
    equal_ranges[index]=unchanged;red_ranges[index]=changed

for i,new in changes.items():revise(blocks[i],new,i)

# Bibliography text was unchanged in this revision. Do not inherit old red marks.
reference_heading=next(i for i,b in enumerate(blocks) if txt(b).strip()=='参考文献')
reference_color_blocks=[]
for i in range(reference_heading+1,len(blocks)):
    paragraph=blocks[i]
    if not txt(paragraph).strip():continue
    reference_color_blocks.append(i)
    for run in paragraph.xpath('.//w:r[w:t]',namespaces=NS):
        rp=run.find('w:rPr',NS)
        if rp is None:
            rp=E.Element(q('rPr'));run.insert(0,rp)
        color=rp.find('w:color',NS)
        if color is None:color=E.SubElement(rp,q('color'))
        color.attrib.clear();color.set(q('val'),'000000')
    audit.append({'block':i,'unit':'unchanged reference color','text_unchanged':True,'color':'000000'})
assert len(reference_color_blocks)==36

def update_cell(cell,value,red):
    paragraph=cell.find('w:p',NS);rp=paragraph.find('w:r/w:rPr',NS)
    rp=deepcopy(rp)
    for p in cell.findall('w:p',NS)[1:]:cell.remove(p)
    for r in paragraph.findall('w:r',NS):paragraph.remove(r)
    new_run(paragraph,str(value),rp,red)

table_values={75:pd.read_csv(HERE/'Table2B.csv',dtype=str).values.tolist(),
              115:pd.read_csv(HERE/'Table4.csv',dtype=str).values.tolist()}
for i,values in table_values.items():
    table=blocks[i];rows=table.findall('w:tr',NS)
    oldmap={txt(row.find('w:tc',NS)):row for row in rows[1:]} if i==115 else {}
    for row in rows[1:]:table.remove(row)
    for j,value in enumerate(values):
        template=oldmap.get(value[0],rows[1]) if i==115 else rows[j+1]
        row=deepcopy(template);cells=row.findall('w:tc',NS)
        assert len(cells)==len(value)
        for cell,new in zip(cells,value):
            previous=txt(cell)
            if previous!=new:update_cell(cell,new,True)
        table.append(row)
    audit.append({'block':i,'unit':'English table body','rows':len(values),'after':values})

# Update the original media identities in the new DOCX only, including SVG fallbacks.
media={
    'word/media/image1.png':'Figure1_study_design_bulk12.png','word/media/image2.svg':'Figure1_study_design_bulk12.svg',
    'word/media/image9.png':'Figure4_twelve_subtype_MMR4_LOCO.png','word/media/image10.svg':'Figure4_twelve_subtype_MMR4_LOCO.svg',
    'word/media/image11.png':'Figure5_comparison_bulk12.png','word/media/image12.svg':'Figure5_comparison_bulk12.svg',
    'word/media/revision_selection_S1.png':'SupplementaryFigureS1_current_selection.png',
    'word/media/revision_stability_S2.png':'SupplementaryFigureS2_bulk12_selection_stability.png'}
updated=dict(original)
for target,filename in media.items():updated[target]=(HERE/'figures'/filename).read_bytes()
for index,target in [(21,'word/media/image1.png'),(121,'word/media/image9.png'),(129,'word/media/image11.png'),
                     (146,'word/media/revision_selection_S1.png'),(148,'word/media/revision_stability_S2.png')]:
    w,h=Image.open(HERE/'figures'/media[target]).size
    for extent in blocks[index].xpath('.//wp:extent',namespaces=NS):
        cx=int(extent.get('cx'));cy=round(cx*h/w);extent.set('cy',str(cy))
        for shape in blocks[index].xpath('.//a:xfrm/a:ext',namespaces=NS):shape.set('cx',str(cx));shape.set('cy',str(cy))
updated['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(DEST,'w') as z:
    for info in infos:z.writestr(info,updated[info.filename])
dump=lambda file,obj:file.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
dump(HERE/'revision_audit.json',audit)
dump(HERE/'sentence_style_ranges.json',{'equal':equal_ranges,'red':red_ranges})
dump(HERE/'document_build_manifest.json',{'source_sha256':sha(SOURCE),'output_sha256':sha(DEST),
      'output_filename':DEST.name,'reference_color_blocks':reference_color_blocks,
      'changed_paragraphs':sorted(changes),'changed_tables':sorted(table_values),
      'changed_media':media,'drawing_blocks':[21,121,129,146,148],
      'body_blocks':len(blocks),'formatting':'Retain all source paragraph properties and unchanged sentence run properties'})
print('Built',DEST)
