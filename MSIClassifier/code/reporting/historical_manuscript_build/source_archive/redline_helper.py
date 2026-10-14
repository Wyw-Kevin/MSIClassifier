"""Mark only text changes against the originally attached third manuscript.

Semantic paragraph/table correspondences are explicit so reordered tables and
new analyses do not displace the comparison. Nothing changes except run color
and the run boundaries required to apply that color.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as E

HERE = Path(__file__).resolve().parent
BASE = Path('/home/wyw/.codex/attachments/20a9bbc7-e6cd-4d8c-bb32-37e48981baea/MMR分模态分类器_第三版.docx')
CURRENT = HERE.parent / 'bulk12_loco_manuscript_seed2026_20261013' / 'MMR分模态分类器_第三版_十二亚型与留一队列MMR4_句子标红_参考文献颜色修正.docx'
OUTPUT = HERE / 'MMR分模态分类器_第三版_与原始第三版对比_仅改动内容标红.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
def tag(n): return '{' + W + '}' + n
def text(p): return ''.join(p.xpath('.//w:t/text()', namespaces=NS))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with ZipFile(p) as z: return E.fromstring(z.read('word/document.xml'))

# Current body block -> original third-version body block(s).
# Entries not listed here must be verbatim unchanged or explicitly new below.
PARAGRAPHS = {
    5: [5], 6: [6], 14: [14], 15: [15], 19: [19], 20: [20],
    22: [22], 25: [25], 26: [26], 30: [29], 33: [32], 34: [33],
    35: [34], 37: [36], 39: [38], 43: [42], 45: [44], 48: [47],
    50: [49], 52: [51], 54: [53], 56: [55], 59: [58], 60: [59],
    63: [60], 65: [61], 66: [61], 67: [62], 71: [66], 78: [68],
    81: [71], 83: [72], 88: [75], 89: [76], 90: [77], 91: [78],
    95: [81], 98: [84], 99: [85], 102: [88], 104: [90], 105: [91],
    108: [94], 111: [97], 113: [98], 118: [101], 119: [102, 103],
    122: [105], 126: [109], 127: [110], 132: [115], 135: [118],
    136: [119], 137: [120], 138: [121], 139: [122], 140: [123],
    141: [124], 144: [127],
    **{152 + i: [129 + i] for i in range(36)},
}
NEW_PARAGRAPHS = {28, 61, 62, 64, 72, 74, 76, 82, 84, 86, 92,
                  112, 114, 116, 120, 145, 147, 149}

# Latin terms and numerical values remain whole: a changed 0.9335 is red
# as a complete value. Chinese characters permit precise phrase-level marking.
TOKENS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž_]*(?:[-’'][A-Za-zÀ-ÖØ-öø-ÿĀ-ž_]+)*|[+-]?\d+(?:[.,]\d+)*%?|\s+|.", re.S)

def diff_mask(old, new):
    aa = [m.group() for m in TOKENS.finditer(old)]
    matches = list(TOKENS.finditer(new))
    bb = [m.group() for m in matches]
    mask = [False] * len(new)
    for op, _, _, j, k in SequenceMatcher(None, aa, bb, autojunk=False).get_opcodes():
        if op in ('insert', 'replace'):
            for m in matches[j:k]: mask[m.start():m.end()] = [True] * (m.end() - m.start())
    return mask

def spans(mask):
    out = []
    for i, v in enumerate(mask):
        if v and (i == 0 or not mask[i-1]): out.append([i, i+1])
        elif v: out[-1][1] = i+1
    return out

def cell(table, row, col):
    return table.findall('w:tr', NS)[row].findall('w:tc', NS)[col].find('w:p', NS)

def make_plan(base, current):
    old = base.find('w:body', NS)
    cur = current.find('w:body', NS)
    exact = {}
    for i, p in enumerate(old):
        if p.tag == tag('p'): exact.setdefault(text(p), i)
    plan = []
    def add(p, before, location, sources, kind):
        after = text(p)
        plan.append({'p': p, 'location': location, 'source': sources,
                     'kind': kind, 'before': before, 'after': after,
                     'mask': diff_mask(before, after)})
    for i, node in enumerate(cur):
        if node.tag != tag('p'): continue
        after = text(node)
        if i in PARAGRAPHS:
            ids = PARAGRAPHS[i]
            add(node, '\n'.join(text(old[k]) for k in ids), f'正文段落{i}', ids, 'paragraph')
        elif after in exact:
            add(node, after, f'正文段落{i}', [exact[after]], 'unchanged')
        else:
            assert i in NEW_PARAGRAPHS, (i, after)
            add(node, '', f'正文段落{i}', [], 'new paragraph')

    # Table 1: retain the same cohorts and compare corresponding columns.
    for r in range(13):
        for c in range(6):
            bc = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}.get(c)
            add(cell(cur[29], r, c), text(cell(old[28], r, bc)) if bc is not None else '',
                f'表1/行{r}/列{c}', [28, r, bc] if bc is not None else [], 'table cell')

    # Table 2A retains the six scRNA results from the old combined Table 2.
    for r in range(7):
        for c in range(5):
            br = 0 if r == 0 else r + 2
            bc = c if r == 0 or c >= 2 else 1
            add(cell(cur[73], r, c), text(cell(old[67], br, bc)),
                f'表2A/行{r}/列{c}', [67, br, bc], 'table cell')

    # Tables 2B and 2C are newly added analyses. An unchanged GEO subset name
    # is carried forward from the original results table rather than marked red.
    old_geo_name = text(cell(old[67], 2, 2))
    for bi, name in [(75, '表2B'), (77, '表2C')]:
        for r, row in enumerate(cur[bi].findall('w:tr', NS)):
            for c, tc in enumerate(row.findall('w:tc', NS)):
                p = tc.find('w:p', NS)
                before = old_geo_name if text(p) == old_geo_name else ''
                add(p, before, f'{name}/行{r}/列{c}', [67, 2, 2] if before else [], 'new table cell')

    # Table 3: same 18 reference subtypes, with the original direction/AUROC
    # columns moved to their new locations and an added consistency column.
    for r in range(19):
        for c in range(5):
            bc = {0: 0, 1: 3, 2: 4, 3: 5}.get(c)
            add(cell(cur[85], r, c), text(cell(old[73], r, bc)) if bc is not None else '',
                f'表3/行{r}/列{c}', [73, r, bc] if bc is not None else [], 'table cell')

    # Table 4: compare the seven retained bulk subtypes by subtype ID, not row
    # position. Four newly selected subtypes and new columns are additions.
    old_rows = {text(cell(old[99], r, 0)): r for r in range(1, 11)}
    for r in range(13):
        for c in range(6):
            br = 0 if r == 0 else old_rows.get(text(cell(cur[115], r, 0)))
            bc = {0: 0, 1: 4, 3: 5, 4: 6}.get(c)
            before = text(cell(old[99], br, bc)) if br is not None and bc is not None else ''
            add(cell(cur[115], r, c), before, f'表4/行{r}/列{c}',
                [99, br, bc] if before else [], 'table cell')
    assert len(plan) == len(current.xpath('//w:body//w:p', namespaces=NS))
    return plan

def color_run(run, is_red):
    rp = run.find('w:rPr', NS)
    if rp is None:
        rp = E.Element(tag('rPr'))
        run.insert(0, rp)
    color = rp.find('w:color', NS)
    if color is None:
        color = E.Element(tag('color'))
        # Keep rPr child ordering consistent with Word's schema.
        later = {tag(n) for n in ('spacing', 'w', 'kern', 'position', 'sz', 'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText', 'vertAlign', 'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath', 'rPrChange')}
        index = next((i for i, n in enumerate(rp) if n.tag in later), len(rp))
        rp.insert(index, color)
    color.attrib.clear()
    color.set(tag('val'), 'FF0000' if is_red else '000000')

def apply_mask(p, mask):
    offset = 0
    for run in list(p.xpath('.//w:r', namespaces=NS)):
        if not run.findall('w:t', NS): continue
        parent = run.getparent()
        replacement = []
        active = None
        active_red = None
        def start(red):
            nonlocal active, active_red
            active = E.Element(run.tag, dict(run.attrib), nsmap=run.nsmap)
            rp = run.find('w:rPr', NS)
            if rp is not None: active.append(deepcopy(rp))
            color_run(active, red)
            active_red = red
            replacement.append(active)
        for node in run:
            if node.tag == tag('rPr'): continue
            if node.tag != tag('t'):
                if active is None: start(False)
                active.append(deepcopy(node))
                continue
            value = node.text or ''
            j = 0
            while j < len(value):
                red = mask[offset + j]
                k = j + 1
                while k < len(value) and mask[offset + k] == red: k += 1
                if active is None or active_red != red: start(red)
                piece = deepcopy(node)
                piece.text = value[j:k]
                if piece.text[:1].isspace() or piece.text[-1:].isspace(): piece.set(XML_SPACE, 'preserve')
                active.append(piece)
                j = k
            if not value:
                if active is None: start(False)
                active.append(deepcopy(node))
            offset += len(value)
        index = parent.index(run)
        for n in replacement:
            parent.insert(index, n)
            index += 1
        parent.remove(run)
    assert offset == len(mask)

def actual_mask(p):
    result = []
    for run in p.xpath('.//w:r', namespaces=NS):
        colors = run.xpath('./w:rPr/w:color/@w:val', namespaces=NS)
        red = bool(colors and colors[0].upper() == 'FF0000')
        result.extend([red] * len(''.join(run.xpath('./w:t/text()', namespaces=NS))))
    assert len(result) == len(text(p))
    return result

def style_trace(p):
    out = []
    for run in p.xpath('.//w:r', namespaces=NS):
        rp = deepcopy(run.find('w:rPr', NS))
        if rp is not None:
            for c in rp.findall('w:color', NS): rp.remove(c)
        style = E.tostring(rp, method='c14n') if rp is not None and len(rp) else b''
        for ch in run:
            if ch.tag == tag('rPr'): continue
            if ch.tag == tag('t'):
                out.extend(('text', v, style, tuple(sorted(run.attrib.items()))) for v in ch.text or '')
            else:
                out.append(('other', E.tostring(ch, method='c14n'), style, tuple(sorted(run.attrib.items()))))
    return out

def verify(base, before, after, plan):
    bp = before.xpath('//w:body//w:p', namespaces=NS)
    ap = after.xpath('//w:body//w:p', namespaces=NS)
    assert len(bp) == len(ap) == len(plan)
    # Plan order has main paragraphs followed by tables, so compare by XPath.
    by_path = {before.getroottree().getpath(x['p']): x for x in plan}
    red_chars = 0
    for a, b in zip(bp, ap):
        entry = by_path[before.getroottree().getpath(a)]
        assert text(a) == text(b)
        assert style_trace(a) == style_trace(b), entry['location']
        assert E.tostring(a.find('w:pPr', NS)) == E.tostring(b.find('w:pPr', NS)) if a.find('w:pPr', NS) is not None else b.find('w:pPr', NS) is None
        assert actual_mask(b) == entry['mask'], entry['location']
        red_chars += sum(entry['mask'])
        if entry['before'] == entry['after']: assert not any(actual_mask(b))
    for xp in ('//w:tblPr', '//w:tblGrid', '//w:trPr', '//w:tcPr', '//w:sectPr'):
        assert [E.tostring(n, method='c14n') for n in before.xpath(xp, namespaces=NS)] == [E.tostring(n, method='c14n') for n in after.xpath(xp, namespaces=NS)]
    mathns = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    assert [E.tostring(n, method='c14n') for n in before.xpath('//m:oMath', namespaces=mathns)] == [E.tostring(n, method='c14n') for n in after.xpath('//m:oMath', namespaces=mathns)]
    with ZipFile(CURRENT) as a, ZipFile(OUTPUT) as b:
        assert a.namelist() == b.namelist()
        for name in a.namelist():
            if name != 'word/document.xml': assert a.read(name) == b.read(name), name
        assert b.testzip() is None
    # Independent checks on representative unchanged passages, figures and
    # original numerical results guard against wholesale coloring.
    body = after.find('w:body', NS)
    for i in (0, 1, 4, 12, 13, 38, 40, 87, 96, 97, 100, 103, 125, 131, 178):
        assert not any(actual_mask(body[i])), i
    for r in range(1, 19):
        for c in range(4): assert not any(actual_mask(cell(body[85], r, c)))
    # Replacing the label 阳性 with MSI must not color the unchanged count 35.
    for term in ('MSI35', 'MSS29', 'MSI4', 'MSS19', 'MSI2', 'MSS13'):
        value, mask = text(body[25]), actual_mask(body[25])
        start = value.index(term)
        digit = start + next(i for i, ch in enumerate(term) if ch.isdigit())
        assert all(mask[start:digit]) and not any(mask[digit:start + len(term)])
    return {'status': 'passed', 'baseline': str(BASE), 'baseline_sha256': sha(BASE),
            'input': str(CURRENT), 'input_sha256': sha(CURRENT), 'output': str(OUTPUT),
            'output_sha256': sha(OUTPUT), 'paragraphs_checked': len(ap),
            'red_characters_before': sum(sum(actual_mask(x)) for x in bp),
            'red_characters_after': red_chars, 'text_preserved': True,
            'noncolor_character_styles_preserved': True,
            'paragraph_and_table_layout_preserved': True,
            'equations_figures_and_other_package_parts_preserved': True,
            'unchanged_paragraphs_and_cells_black': sum(x['before'] == x['after'] and bool(x['after']) for x in plan)}

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--check-input', action='store_true')
    opt = args.parse_args()
    base, current = read(BASE), read(CURRENT)
    plan = make_plan(base, current)
    if opt.check_input:
        false_red = sum(sum(x and not y for x, y in zip(actual_mask(p['p']), p['mask'])) for p in plan)
        missing_red = sum(sum(y and not x for x, y in zip(actual_mask(p['p']), p['mask'])) for p in plan)
        print(json.dumps({'false_red_characters': false_red, 'missing_change_mark_characters': missing_red}))
        raise SystemExit(1 if false_red or missing_red else 0)
    before = deepcopy(current)
    before_plan = make_plan(base, before)
    audit = []
    for item in plan:
        apply_mask(item['p'], item['mask'])
        audit.append({k: v for k, v in item.items() if k not in ('p', 'mask')} | {'red_ranges': spans(item['mask'])})
    xml = E.tostring(current, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(CURRENT) as src, ZipFile(OUTPUT, 'w') as out:
        for entry in src.infolist(): out.writestr(entry, xml if entry.filename == 'word/document.xml' else src.read(entry.filename))
    result = verify(base, before, read(OUTPUT), before_plan)
    (HERE / 'redline_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (HERE / 'content_difference_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    lines = ['与原始第三版的内容差异核对', '', f'对比基准：{BASE.name}', f'当前内容来源：{CURRENT.name}',
             '本次仅调整修改标记；不改动当前稿件的文字、数值、图片、公式及版式。',
             '红色表示相对原始第三版实际新增或替换的词句、数值及标点。相同文字及仅移动位置的已有表格数值恢复黑色。',
             '删除的旧内容不出现在当前稿件；下列对照保留旧表述供核查。新增段落或表格内容按新增内容标红。',
             '参考文献按同一序号逐条比较，仅补全的作者、来源及DOI等变动部分标红。', '',
             f'原稿红色字符数：{result["red_characters_before"]}；核对后红色字符数：{result["red_characters_after"]}。', '']
    for item in audit:
        if item['before'] == item['after']: continue
        marked = item['after']
        for start, end in reversed(item['red_ranges']): marked = marked[:start] + '【' + marked[start:end] + '】' + marked[end:]
        lines.extend([item['location'], '原第三版：' + (item['before'] or '（新增内容）'), '当前稿件：' + marked, ''])
    (HERE / '第三版差异核对说明.txt').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
