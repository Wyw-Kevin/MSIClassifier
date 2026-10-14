"""Local evidence amendments; precise original-third-version redline, refs black."""
from pathlib import Path
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib, importlib.util, json, re
from zipfile import ZipFile
from lxml import etree as E
from PIL import Image
from io import BytesIO

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source_archive/MMR分模态分类器_第三版_修订.docx'
DEST = HERE / 'MMR分模态分类器_第三版_代码核实与配对增量修订_仅改动标红.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q = lambda n: '{' + NS['w'] + '}' + n
txt = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon = lambda p: E.tostring(p, method='c14n')
spec = importlib.util.spec_from_file_location('baseline_diff', HERE / 'source_archive/redline_helper.py')
diff = importlib.util.module_from_spec(spec); spec.loader.exec_module(diff)
diff.BASE = HERE / 'source_archive/original_third_version.docx'

with ZipFile(SOURCE) as z:
    infos = z.infolist(); package = {v.filename: z.read(v.filename) for v in infos}
root = E.fromstring(package['word/document.xml']); original = deepcopy(root)
body = root.find('w:body', NS)
changes = json.loads((HERE / 'text_changes.json').read_text())
audit = []

def replace_paragraph(p, value):
    before = txt(p)
    style = []
    for run in p.findall('w:r', NS):
        assert all(c.tag in {q('rPr'), q('t')} for c in run)
        props = run.find('w:rPr', NS)
        props = deepcopy(props) if props is not None else E.Element(q('rPr'))
        assert props.find('w:sz', NS).get(q('val')) == '24'
        for ch in ''.join(run.xpath('./w:t/text()', namespaces=NS)):
            style.append((props, dict(run.attrib)))
    assert len(style) == len(before) and style
    after_styles = []
    for op, i, j, a, b in SequenceMatcher(None, before, value, autojunk=False).get_opcodes():
        if op == 'equal': after_styles.extend(style[i:j])
        elif op in ('replace', 'insert'):
            template = style[min(i, len(style)-1)]
            after_styles.extend([template] * (b-a))
    assert len(after_styles) == len(value)
    for run in list(p.findall('w:r', NS)): p.remove(run)
    index = 0
    while index < len(value):
        props, attrs = after_styles[index]
        key = canon(props), attrs
        stop = index + 1
        while stop < len(value) and (canon(after_styles[stop][0]), after_styles[stop][1]) == key: stop += 1
        run = E.SubElement(p, q('r'), attrs); run.append(deepcopy(props))
        node = E.SubElement(run, q('t')); node.text = value[index:stop]
        if node.text[:1].isspace() or node.text[-1:].isspace(): node.set(diff.XML_SPACE, 'preserve')
        index = stop
    assert txt(p) == value

for key, value in changes['paragraphs'].items():
    i = int(key); before = txt(body[i])
    pp = body[i].find('w:pPr', NS); props = canon(pp) if pp is not None else None
    replace_paragraph(body[i], value)
    assert props == (canon(body[i].find('w:pPr', NS)) if pp is not None else None)
    audit.append({'block': i, 'before': before, 'after': value})
new_ref = deepcopy(body[187]); replace_paragraph(new_ref, changes['new_reference'])
body.insert(188, new_ref)
before_color = deepcopy(root)

# Cumulative changes are compared with the originally attached third version;
# bibliography is explicitly excluded per the user's subsequent preference.
diff.NEW_PARAGRAPHS.add(188)
baseline = diff.read(diff.BASE)
# The user also revised passages that had been unchanged in the prior delivery
# (including the title/introduction). Carry forward their known original block
# correspondences instead of treating them as wholly new paragraphs.
old_body = baseline.find('w:body', NS)
exact_original = {txt(p): i for i, p in enumerate(old_body) if p.tag == q('p')}
prior_inventory = json.loads((HERE / 'source_archive/prior_current_inventory.json').read_text())
for item in prior_inventory:
    if 'text' in item and item['block'] not in diff.PARAGRAPHS and item['block'] not in diff.NEW_PARAGRAPHS:
        if item['text'] in exact_original: diff.PARAGRAPHS[item['block']] = [exact_original[item['text']]]
plan = diff.make_plan(baseline, root)
reference_blocks = set(range(151, 189))
redline_audit = []
for entry in plan:
    path = root.getroottree().getpath(entry['p'])
    if entry['location'].startswith('正文段落') and int(entry['location'][4:]) in reference_blocks:
        entry['mask'] = [False] * len(entry['after'])
    diff.apply_mask(entry['p'], entry['mask'])
    assert diff.actual_mask(entry['p']) == entry['mask'], entry['location']
    redline_audit.append({'location': entry['location'], 'source': entry['source'],
                          'after': entry['after'], 'red_ranges': diff.spans(entry['mask'])})
for p, a in zip(before_color.xpath('//w:body//w:p', namespaces=NS), root.xpath('//w:body//w:p', namespaces=NS)):
    assert txt(p) == txt(a) and diff.style_trace(p) == diff.style_trace(a)

labels = json.loads((HERE / 'figure_label_manifest.json').read_text())
assert labels['verification_status'] == 'passed'
replaced_media = {}
for key, path in labels['embedded_media_replacement_mapping'].items():
    name = 'word/media/' + key + '.png'
    assert name in package
    record = next(x for x in labels['files'] if x['media_key'] == key)
    assert hashlib.sha256(package[name]).hexdigest() == record['source_png_sha256']
    new = Path(path).read_bytes()
    assert Image.open(BytesIO(new)).size == Image.open(BytesIO(package[name])).size
    replaced_media[name] = new
xml = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(DEST, 'w') as z:
    for entry in infos:
        value = xml if entry.filename == 'word/document.xml' else replaced_media.get(entry.filename, package[entry.filename])
        z.writestr(entry, value)

with ZipFile(DEST) as z:
    assert z.testzip() is None and z.namelist() == list(package)
    final = E.fromstring(z.read('word/document.xml'))
    fb = final.find('w:body', NS); ob = original.find('w:body', NS)
    expected_changes = {int(k): v for k, v in changes['paragraphs'].items()}
    for i in range(188):
        assert txt(fb[i]) == expected_changes.get(i, txt(ob[i])), i
        pp = ob[i].find('w:pPr', NS); qp = fb[i].find('w:pPr', NS)
        assert (canon(pp) if pp is not None else None) == (canon(qp) if qp is not None else None), i
    assert txt(fb[188]) == changes['new_reference']
    for i in range(152, 189):
        assert not any(diff.actual_mask(fb[i])), i
        assert fb[i].xpath('.//w:r[w:t]/w:rPr/w:color/@w:val', namespaces=NS)
    # Main tables retain every English string and all table/cell layout.
    for xp in ('//w:tbl//w:t/text()', '//w:tblPr', '//w:tblGrid', '//w:trPr', '//w:tcPr', '//w:sectPr', '//m:oMath'):
        a, b = original.xpath(xp, namespaces=NS), final.xpath(xp, namespaces=NS)
        assert [canon(n) if isinstance(n, E._Element) else str(n) for n in a] == [canon(n) if isinstance(n, E._Element) else str(n) for n in b], xp
    assert all(re.search(r'[\u4e00-\u9fff]', value) is None for value in final.xpath('//w:tbl//w:t/text()', namespaces=NS))
    for name, value in package.items():
        if name != 'word/document.xml' and name not in replaced_media: assert z.read(name) == value, name
    for name, value in replaced_media.items(): assert z.read(name) == value
    for i in expected_changes:
        if i < 151:
            assert set(fb[i].xpath('.//w:r[w:t]/w:rPr/w:sz/@w:val', namespaces=NS)) == {'24'}
            assert fb[i].xpath('./w:pPr/w:spacing/@w:line', namespaces=NS) == ['480']
            assert fb[i].xpath('./w:pPr/w:spacing/@w:lineRule', namespaces=NS) == ['exact']
    full = txt(fb)
    assert 'bulk12' not in full and '加入MSH2、MSH6和PMS2未提高' not in full
    for term in ('0.0315（95% CI 0.0102–0.0566）', '0.0335（0.0118–0.0599）',
                 'pybayesprism 0.1.0[37]', '12,079', '不先进行患者内汇总', '字典序最小值'):
        assert term in full, term

(HERE / 'revision_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
(HERE / 'redline_audit.json').write_text(json.dumps(redline_audit, ensure_ascii=False, indent=2))
result = {'status': 'passed', 'source': str(SOURCE), 'source_sha256': sha(SOURCE),
          'baseline_redline_source': str(diff.BASE), 'output': str(DEST), 'output_sha256': sha(DEST),
          'locally_revised_paragraphs': len(changes['paragraphs']), 'references_black': 37,
          'figures_label_only_replaced': len(replaced_media), 'all_existing_table_strings_preserved': True,
          'paragraph_layout_fonts_and_24pt_spacing_preserved': True,
          'mathematical_equations_and_unrelated_package_parts_preserved': True,
          'actual_changed_text_only_red': True, 'reference_model_convergence_limit_retained': True}
(HERE / 'manuscript_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
