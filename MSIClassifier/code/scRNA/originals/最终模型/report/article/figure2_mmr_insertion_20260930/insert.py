from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import re,json,shutil,cairosvg,hashlib
A=Path('/home/wyw/rstudio/MSI/最终模型/report/article');O=A/'figure2_mmr_insertion_20260930';p=A/'MMR多转录组分类器_第二版.docx'
S='http://www.w3.org/2000/svg';W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';ns={'w':W}
with ZipFile(p) as z:infos=z.infolist();d={i.filename:z.read(i.filename) for i in infos}
svg=E.fromstring(d['word/media/image6.svg']);original=deepcopy(svg);width=514.6;shift=400;cut=480;height=1048
svg.set('viewBox',f'0 0 {width} {height}');svg.set('height',f'{height}pt')
for id in ['axes_5','axes_6']:
 n=svg.xpath('//*[@id="'+id+'"]')[0];n.set('transform',f'translate(0,{shift})')
for a,b in [('E','I'),('F','J')]:
 n=svg.xpath('//*[@id="panel-label-'+a+'"]')[0];n.text=b;n.set('id','panel-label-'+b);n.set('y',str(float(n.get('y'))+shift))
box=E.fromstring((A/'mmr4_unweighted_boxplots_20260929/MMR4_unweighted_sample_expression.svg').read_bytes());scale=width/756
# Namespace all IDs and references to avoid collision with the existing figure.
ids={n.get('id'):'mmr_'+n.get('id') for n in box.iter() if n.get('id')}
for n in box.iter():
 for k,v in list(n.attrib.items()):
  if k=='id':n.set(k,ids[v]);continue
  v=re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+ids.get(m[1],m[1])+')',v)
  if v.startswith('#') and v[1:] in ids:v='#'+ids[v[1:]]
  n.set(k,v)
for n in list(box.xpath('//*[local-name()="text"]')):
 t=''.join(n.itertext())
 if t in ['A','B','C','D']:
  index='ABCD'.index(t);marker=E.SubElement(svg,'{'+S+'}text',x=str(8 if index%2==0 else 266),y=str(cut+float(n.get('y'))*scale),attrib={'font-family':'DejaVu Sans','font-size':'11.479618738592578','font-weight':'bold','fill':'#20242a','id':'panel-label-'+'EFGH'[index]});marker.text='EFGH'[index]
  n.getparent().remove(n)
g=E.SubElement(svg,'{'+S+'}g',transform=f'translate(0,{cut}) scale({scale})',id='four-gene-expression-panels')
for n in list(box):g.append(n)
# Ensure the whole expanded canvas is white.
bg=E.Element('{'+S+'}rect',x='0',y='0',width=str(width),height=str(height),fill='white');svg.insert(0,bg)
for marker in list(svg.xpath('//*[starts-with(@id,"panel-label-")]')):
 marker.getparent().remove(marker);svg.append(marker)
assert sorted(n.text for n in svg.xpath('//*[starts-with(@id,"panel-label-")]'))==list('ABCDEFGHIJ')
svgb=E.tostring(svg,encoding='UTF-8',xml_declaration=True);(O/'Figure02.svg').write_bytes(svgb)
cairosvg.svg2png(bytestring=svgb,write_to=str(O/'Figure02.png'),output_width=1800);cairosvg.svg2pdf(bytestring=svgb,write_to=str(O/'Figure02.pdf'))
parts=[]
for suffix,y,h in [('ABCD',0,cut),('EFGHIJ',cut,height-cut)]:
 s=deepcopy(svg);s.set('viewBox',f'0 {y} {width} {h}');s.set('height',f'{h}pt');blob=E.tostring(s,encoding='UTF-8',xml_declaration=True)
 (O/f'Figure02_{suffix}.svg').write_bytes(blob);png=cairosvg.svg2png(bytestring=blob,output_width=2200);(O/f'Figure02_{suffix}.png').write_bytes(png);parts.append((blob,png,h))
r=E.fromstring(d['word/document.xml']);body=r.find('w:body',ns);changes=[]
imagepara=next(n for n in body if n.xpath('.//*[local-name()="docPr"][@descr="图2"]'))
second=deepcopy(imagepara)
for para,h in [(imagepara,parts[0][2]),(second,parts[1][2])]:
 ex=para.xpath('.//*[local-name()="extent"]')[0];cx=int(ex.get('cx'));cy=round(cx*h/width)
 for n in para.xpath('.//*[local-name()="extent" or local-name()="ext"]'):
  if n.get('cx') is not None and n.get('cy') is not None:n.set('cy',str(cy))
for n in second.xpath('.//*[@r:embed]',namespaces={'r':R}):n.set('{'+R+'}embed','rIdFigure2ContinueSVG' if n.tag.endswith('svgBlip') else 'rIdFigure2ContinuePNG')
maxid=max(int(n.get('id')) for n in r.xpath('//*[local-name()="docPr"]'))
for n in second.xpath('.//*[local-name()="docPr"]'):n.set('id',str(maxid+1));n.set('name','Figure 2 continued');n.set('descr','图2续：E–J')
pr=second.find('w:pPr',ns)
if pr is None:pr=E.SubElement(second,'{'+W+'}pPr')
E.SubElement(pr,'{'+W+'}pageBreakBefore')
body.insert(body.index(imagepara)+1,second)
def put(n,new):
 ts=n.xpath('.//w:t',namespaces=ns);old=''.join(t.text or '' for t in ts);ts[0].text=new
 for t in ts[1:]:t.text=''
 changes.append({'before':old,'after':new})
for n in list(body.findall('w:p',ns)):
 t=''.join(n.xpath('.//w:t/text()',namespaces=ns));new=t.replace('图2E','图2I').replace('图2F','图2J')
 if t.startswith('图2.'):new=new.replace('C–F采用当前MLH1定义及Z-score，随机种子同为2026。','C–D、I–J为当前单细胞分析；E–H展示未做比例加权的四个MMR基因表达。分类模型采用Z-score，显式随机种子为2026。')
 if t.startswith('（E）三个scRNA'):new='（I）'+new[3:]
 if t.startswith('（F）当前加权'):new='（J）'+new[3:]
 if t.startswith('为同时表征候选恶性上皮中的MLH1表达'):
  new='首先分别展示MLH1、MSH2、MSH6和PMS2在候选恶性上皮中的样本级平均表达，按队列比较MSI与MSS分布（图2E–H）；该分析不进行比例加权，用于描述四基因表达而非检验增加基因的预测增量。'+new
 if new!=t:put(n,new)
 if t.startswith('（D）18亚型比例'):
  cap=deepcopy(n);put(cap,'（E–H）依次为MLH1、MSH2、MSH6和PMS2在三个单细胞队列中的样本级表达分布。每个队列内并排展示MSS与MSI，散点代表肿瘤样本；纵坐标为同一CNV保留集合中候选恶性上皮细胞的归一化表达均值，不进行比例加权或样本间Z-score。箱体为第25至第75百分位，中线为中位数，须线为1.5倍四分位距内的最远观测值，全部样本以散点展示并保留实测零值。三个队列的MSS/MSI例数分别为29/35、19/4和13/2，各基因采用独立纵轴范围。');n.addnext(cap)
rels=E.fromstring(d['word/_rels/document.xml.rels']);rn=rels.tag.split('}')[0]+'}'
for id,target in [('rIdFigure2ContinuePNG','media/figure2_continued.png'),('rIdFigure2ContinueSVG','media/figure2_continued.svg')]:E.SubElement(rels,rn+'Relationship',Id=id,Type=R+'/image',Target=target)
d['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True);d['word/_rels/document.xml.rels']=E.tostring(rels,encoding='UTF-8',xml_declaration=True,standalone=True)
d['word/media/image6.svg']=parts[0][0];d['word/media/image5.png']=parts[0][1];d['word/media/figure2_continued.svg']=parts[1][0];d['word/media/figure2_continued.png']=parts[1][1]
backup=O/'MMR多转录组分类器_第二版_插入四基因图前.docx';assert not backup.exists();shutil.copy2(p,backup)
with ZipFile(O/'updated.docx','w') as z:
 for info in infos:z.writestr(info,d.pop(info.filename))
 for name,data in d.items():z.writestr(name,data)
with ZipFile(O/'updated.docx') as z:
 assert z.testzip() is None
 for name in z.namelist():
  if name.endswith(('.xml','.rels','.svg')):E.fromstring(z.read(name))
 with ZipFile(backup) as a:
  ar=E.fromstring(a.read('word/document.xml'))
  assert [E.tostring(n) for n in ar.xpath('//w:tbl',namespaces=ns)]==[E.tostring(n) for n in r.xpath('//w:tbl',namespaces=ns)]
  assert [E.tostring(n) for n in ar.xpath('//*[local-name()="oMath"]')]==[E.tostring(n) for n in r.xpath('//*[local-name()="oMath"]')]
shutil.move(O/'updated.docx',p)
(O/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2));print('Saved figure 2 A-J; Word uses A-D and E-J continuation images to preserve readable size.')
