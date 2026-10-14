"""Fresh cross-artifact checks on saved DOCX, workbook and paired distributions."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import hashlib, json, re
import numpy as np
import pandas as pd
import openpyxl
from lxml import etree as E
from PIL import Image
from io import BytesIO

HERE = Path(__file__).resolve().parent
q = lambda n: '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + n
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math', 's': 'http://www.w3.org/2000/svg'}
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = []
def check(test, description):
    assert test, description
    checks.append(description)

manifest = json.loads((HERE / 'manuscript_verification.json').read_text())
check(manifest['status'] == 'passed', 'DOCX build verified')
doc = Path(manifest['output']); check(sha(doc) == manifest['output_sha256'], 'Saved DOCX hash')
changes = json.loads((HERE / 'text_changes.json').read_text())
with ZipFile(doc) as z, ZipFile(manifest['source']) as source:
    check(z.testzip() is None, 'Word package integrity')
    a = E.fromstring(z.read('word/document.xml')); b = E.fromstring(source.read('word/document.xml'))
    body, before = a.find('w:body', NS), b.find('w:body', NS)
    check(text(body[188]) == changes['new_reference'], 'Added implementation reference37')
    for i in range(188):
        check(text(body[i]) == changes['paragraphs'].get(str(i), text(before[i])), f'Expected text block{i}')
        check(E.tostring(body[i].find('w:pPr', NS)) == E.tostring(before[i].find('w:pPr', NS))
              if before[i].find('w:pPr', NS) is not None else body[i].find('w:pPr', NS) is None, f'Paragraph layout block{i}')
    check(len(body.xpath('./w:p[position()>1]', namespaces=NS)) > 0, 'Main text readable')
    for i in range(152, 189):
        check(set(body[i].xpath('.//w:r[w:t]/w:rPr/w:color/@w:val', namespaces=NS)) == {'000000'}, f'Reference{i-151} black')
    for xp in ['//m:oMath', '//w:tblPr', '//w:tblGrid', '//w:tcPr']:
        check([E.tostring(x, method='c14n') for x in a.xpath(xp, namespaces=NS)] == [E.tostring(x, method='c14n') for x in b.xpath(xp, namespaces=NS)], 'Unchanged ' + xp)
    check(a.xpath('//w:tbl//w:t/text()', namespaces=NS) == b.xpath('//w:tbl//w:t/text()', namespaces=NS), 'Main table strings unchanged')
    for value in a.xpath('//w:tbl//w:t/text()', namespaces=NS): check(re.search(r'[\u4e00-\u9fff]', value) is None, 'Main table English')
    labels = json.loads((HERE / 'figure_label_manifest.json').read_text())
    for rec in labels['files']:
        key = 'word/media/' + rec['media_key'] + '.png'
        check(z.read(key) == Path(rec['output_png']).read_bytes(), 'Embedded revised figure ' + key)
        check(Image.open(BytesIO(z.read(key))).size == Image.open(BytesIO(source.read(key))).size, 'Preserved image dimensions ' + key)
        old, new = E.parse(rec['source_svg']), E.parse(rec['output_svg'])
        def without_text(tree):
            node = deepcopy(tree.getroot())
            for t in node.xpath('//s:text', namespaces=NS): t.getparent().remove(t)
            return E.tostring(node, method='c14n')
        check(without_text(old) == without_text(new), 'Nontext SVG/data geometry unchanged ' + rec['media_key'])
    full = text(body)
    check('历史' in full and '参考模型的逐轮训练历史未保存' in full, 'Reference training-history limitation retained')
    check('加入MSH2、MSH6和PMS2未提高' not in full, 'Unsupported gene-increment assertion removed')
    check('bulk12' not in full, 'bulk-number spacing corrected')

book_path = HERE / 'supplement/SupplementaryTablesS1-S14_English.xlsx'
book = openpyxl.load_workbook(book_path, read_only=True, data_only=True)
old_book = openpyxl.load_workbook(HERE.parent / 'bulk12_loco_manuscript_seed2026_20261013/supplement/SupplementaryTablesS1-S14_English.xlsx', read_only=True, data_only=True)
for name in old_book.sheetnames:
    old_rows = list(old_book[name].values); new_rows = list(book[name].values)
    check(new_rows[:len(old_rows)] == old_rows, 'Existing supplementary sheet preserved ' + name)
for w in book:
    for row in w.values:
        for value in row: check(not isinstance(value, str) or re.search(r'[\u4e00-\u9fff]', value) is None, 'Supplement cells English')

cohort = pd.read_csv(HERE / 'paired_increment/S13_fusion_vs_MMR4_cohort.csv', float_precision='round_trip')
macro = pd.read_csv(HERE / 'paired_increment/S13_fusion_vs_MMR4_macro.csv', float_precision='round_trip')
dist = np.load(HERE / 'paired_increment/bootstrap_AUROC_distributions.npz')
for row in cohort.itertuples():
    delta = dist[row.dataset + '__Fusion'] - dist[row.dataset + '__MMR4']
    check(np.array_equal(delta, dist[row.dataset + '__Delta']), 'Component-paired bootstrap ' + row.dataset)
    lo, hi = np.quantile(delta, [.025, .975])
    check(abs(lo-row.CI_low) < 1e-15 and abs(hi-row.CI_high) < 1e-15, 'Cohort interval ' + row.dataset)
for row in macro.itertuples():
    cs = row.cohorts.split(';')
    values = np.mean([dist[c + '__Delta'] for c in cs], axis=0)
    lo, hi = np.quantile(values, [.025, .975])
    check(abs(lo-row.CI_low) < 1e-15 and abs(hi-row.CI_high) < 1e-15, 'Macro interval ' + row.scope)
    check(abs(row.Delta_AUROC_Fusion_minus_MMR4 - (row.Fusion_macro_AUROC-row.MMR4_macro_AUROC)) < 1e-15, 'Unrounded macro delta ' + row.scope)
for sheet, frame in [('S13_Fusion_vs_MMR4', cohort), ('S13_Fusion_macro', macro)]:
    saved = pd.DataFrame(list(book[sheet].values)[1:], columns=list(book[sheet].values)[0])
    check(saved.columns.tolist() == frame.columns.tolist() and len(saved) == len(frame), 'Paired workbook shape ' + sheet)
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]): check(np.allclose(saved[column].astype(float), frame[column], rtol=0, atol=1e-14), 'Paired workbook numbers ' + sheet + '/' + column)
        else: check(saved[column].tolist() == frame[column].tolist(), 'Paired workbook labels ' + sheet + '/' + column)
check('0.0315（95% CI 0.0102–0.0566）' in full and '0.0335（0.0118–0.0599）' in full, 'Body paired intervals match saved data')
with ZipFile(HERE / 'supplement/SupplementaryMethods_English.docx') as z:
    check(z.testzip() is None, 'Supplementary methods Word package integrity')
    check(all(E.fromstring(z.read(n)) is not None for n in z.namelist() if n.endswith(('.xml', '.rels'))), 'Supplementary methods XML well formed')
check(json.loads((HERE / 'audit_mmr4/verification.json').read_text())['n_values_verified'] == 4848, 'MMR4 original-scale replay complete')
spatial = json.loads((HERE / 'audit_spatial/saved_spatial_replay_summary.json').read_text())
check(spatial['passed'] and spatial['n_sections'] == 14 and spatial['n_spots'] == 20733, 'Spatial count/posterior replay complete')
sc = json.loads((HERE / 'audit_scrna/verification.json').read_text())
check(sc['sample_count'] == 64 and sc['patient_count'] == 62 and sc['S7_workbook_csv_and_score_replay_match'], 'scRNA units and controls audited')
check(json.loads((HERE / 'paired_increment/verification.json').read_text())['status'] == 'passed', 'Paired independent numeric verification passed')
replay = Path('/tmp/msi_method_evidence_paired_replay_20261014')
for name in ['S13_fusion_vs_MMR4_cohort.csv', 'S13_fusion_vs_MMR4_macro.csv', 'paired_sample_predictions.csv']:
    check((replay / name).read_bytes() == (HERE / 'paired_increment' / name).read_bytes(), 'Archived standalone replay identical ' + name)
result = {'status': 'passed', 'checks': len(checks), 'manuscript': str(doc), 'manuscript_sha256': sha(doc),
          'workbook_sha256': sha(book_path), 'references_black': 37,
          'paired_samples': 1212, 'paired_cohorts': 8, 'resamples': 2000,
          'existing_supplementary_tables_preserved': True, 'new_macro_intervals': macro.to_dict('records')}
(HERE / 'delivery_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('PASS final delivery:', len(checks), 'saved artifact checks')
