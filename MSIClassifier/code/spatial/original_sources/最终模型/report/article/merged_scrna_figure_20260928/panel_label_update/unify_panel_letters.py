from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import cairosvg,json,re,hashlib,shutil,base64
A=Path('/home/wyw/rstudio/MSI/最终模型/report/article');O=A/'merged_scrna_figure_20260928';P=O/'panel_label_update';P.mkdir(exist_ok=True)
f=A/'MMR多转录组分类器_第二版.docx';backup=P/'MMR多转录组分类器_第二版_分面标注调整前.docx';assert not backup.exists()
with ZipFile(f) as z:infos=z.infolist();blobs={i.filename:z.read(i.filename) for i in infos}
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';S='http://www.w3.org/2000/svg';ns={'w':W};r=E.fromstring(blobs['word/document.xml'])
parts=[('image2.svg','image1.png',1),('image6.svg','image5.png',2),('image8.svg','image7.png',3),('image10.svg','image9.png',4),('image12.svg','image11.png',5)]
audit=[]
for svgname,pngname,num in parts:
 svg=E.fromstring(blobs['word/media/'+svgname]);vb=[float(v) for v in svg.get('viewBox').split()]
 p=next(p for p in r.xpath('//w:body/w:p',namespaces=ns) if p.xpath('.//*[local-name()="docPr"][@descr="图'+str(num)+'"]'))
 extent=p.xpath('.//*[local-name()="extent"]')[0];widthpt=int(extent.get('cx'))/12700
 # Extra right margin protects merged heatmap colorbar tick labels.
 if num==2:
  vb[2]+=25;svg.set('viewBox',' '.join(map(str,vb)));svg.set('width',str(vb[2])+'pt')
  newcy=round(int(extent.get('cx'))*vb[3]/vb[2])
  for n in p.xpath('.//*[local-name()="extent" or local-name()="ext"]'):
   if n.get('cx') is not None and n.get('cy') is not None:n.set('cy',str(newcy))
 fs=11*vb[2]/widthpt
 found=[]
 for n in list(svg.xpath('//*[local-name()="text"]')):
  old=''.join(n.itertext());m=re.match(r'^([A-Fa-f])(?:\s{1,}(.*))?$',old)
  if not m:continue
  letter=m[1].upper();title=m[2];x=float(n.get('x','0'));y=float(n.get('y','0'))
  if title:
   n.text=title
   for ch in list(n):n.remove(ch)
   marker=E.SubElement(svg,'{'+S+'}text');marker.text=letter
   if num==1:
    mx=30 if letter=='B' else 610;my=y;newx=75 if letter=='B' else 655
    n.set('x',str(newx));n.attrib.pop('transform',None)
   elif num==2:mx=8 if letter in ['A','B','C','E'] else 266;my=y
   elif num==3:
    mx=8 if letter=='A' else 785;my=max(y,fs+1)
    n.set('x','45' if letter=='A' else '835');n.attrib.pop('transform',None)
   elif num==5:mx=425;my=max(y,fs+1)
   else:mx=x-fs*1.8;my=y
  else:
   marker=n;mx=x;my=max(y,fs+1)
   marker.text=letter
   if num==1:
    # Move the independent A away from the separate existing title.
    for t in svg.xpath('//*[local-name()="text"]'):
     if ''.join(t.itertext())=='Single-cell model development':t.set('x','75')
  marker.attrib.clear();marker.set('x',str(mx));marker.set('y',str(my));marker.set('font-family','DejaVu Sans');marker.set('font-size',str(fs));marker.set('font-weight','bold');marker.set('fill','#20242a');marker.set('text-anchor','start');marker.set('id','panel-label-'+letter)
  found.append(letter)
 expected=list('ABCDEF' if num==2 else 'ABCD' if num==4 else 'ABC')
 assert sorted(found)==sorted(expected),(num,found)
 assert not any(re.match(r'^[A-Fa-f]\s+\S', ''.join(n.itertext())) for n in svg.xpath('//*[local-name()="text"]'))
 data=E.tostring(svg,xml_declaration=True,encoding='UTF-8');stem=O/('Figure%02d'%num)
 stem.with_suffix('.svg').write_bytes(data)
 render_data=data
 if num==3:
  # Retain the original spatial raster pixels; render only the edited title bands.
  # The standalone and embedded SVG retain every original vector point.
  overlay=E.Element('{'+S+'}svg',nsmap={None:S},width=str(vb[2]),height=str(vb[3]),viewBox=' '.join(map(str,vb)))
  bg=E.SubElement(overlay,'{'+S+'}image',x='0',y='0',width=str(vb[2]),height=str(vb[3]))
  bg.set('{http://www.w3.org/1999/xlink}href','data:image/png;base64,'+base64.b64encode(blobs['word/media/'+pngname]).decode())
  for x,y,w,h in [(0,0,1152,29),(780,560,372,33)]:E.SubElement(overlay,'{'+S+'}rect',x=str(x),y=str(y),width=str(w),height=str(h),fill='white')
  for n in svg.xpath('//*[local-name()="text"]'):
   if n.get('id','').startswith('panel-label-') or ''.join(n.itertext()) in ['Spatial MLH1 expression and 18-subtype patterns','Section components and model scores','Exploratory spatial transfer']:overlay.append(deepcopy(n))
  render_data=E.tostring(overlay)
 png=cairosvg.svg2png(bytestring=render_data,output_width=2200)
 stem.with_suffix('.png').write_bytes(png);cairosvg.svg2pdf(bytestring=render_data,write_to=str(stem.with_suffix('.pdf')))
 blobs['word/media/'+svgname]=data;blobs['word/media/'+pngname]=png
 if num==2:
  for ext in ['svg','png','pdf']:shutil.copy2(stem.with_suffix('.'+ext),O/('Figure02_model_development_evaluation.'+ext))
 audit.append({'figure':num,'letters':sorted(found),'display_font_pt':11,'letters_separate_from_titles':True})
# Match the workflow caption's A/B/C case to the figure.
for p in r.xpath('//w:body/w:p',namespaces=ns):
 ts=p.xpath('.//w:t',namespaces=ns);text=''.join(t.text or '' for t in ts)
 if text.startswith('图1.'):
  new=text.replace('（a）','（A）').replace('（b）','（B）').replace('（c）','（C）')
  ts[0].text=new
  for t in ts[1:]:t.text=''
blobs['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
shutil.copy2(f,backup);tmp=f.with_suffix('.labels.tmp')
with ZipFile(tmp,'w') as z:
 for i in infos:z.writestr(i,blobs[i.filename])
with ZipFile(tmp) as z:
 assert z.testzip() is None
 for n in z.namelist():
  if n.endswith(('.xml','.rels','.svg')):E.fromstring(z.read(n))
tmp.replace(f)
v=json.loads((O/'verification.json').read_text());v['sha256']=hashlib.sha256(f.read_bytes()).hexdigest();v['panel_labels']=audit
(O/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));shutil.copy2(__file__,P/'unify_panel_letters.py')
print(json.dumps(audit,ensure_ascii=False))
