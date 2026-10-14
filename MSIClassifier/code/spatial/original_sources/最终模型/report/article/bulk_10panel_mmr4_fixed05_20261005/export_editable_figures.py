#!/usr/bin/env python3
"""Export the active manuscript figures as editable SVG/PDF plus preview PNG."""

from __future__ import print_function

import csv
import os
import re
import shutil
import subprocess
from zipfile import ZipFile

ROOT = '/tmp/msi_final'
BRANCH = os.path.join(ROOT, 'report', 'article', 'bulk_10panel_mmr4_fixed05_20261005')
DOCX = '/tmp/active_final.docx'
OUT = os.path.join(ROOT, 'report', 'article', 'editable_main_figures_20261005')
SUPPLEMENT = os.path.join(ROOT, 'report', 'article', 'MMR多转录组分类器_第二版_补充图')

FIGURES = [
    ('Figure01_workflow', 'word/media/image1.png', 'word/media/image2.svg'),
    ('Figure02_scRNA_main', 'word/media/image5.png', 'word/media/image6.svg'),
    ('Figure02_scRNA_continued', 'word/media/figure2_continued.png', 'word/media/figure2_continued.svg'),
    ('Figure03_spatial_transfer', 'word/media/image7.png', 'word/media/image8.svg'),
    ('Figure04_bulk', 'word/media/image9.png', 'word/media/image10.svg'),
    ('Figure05_comparison', 'word/media/image11.png', 'word/media/image12.svg'),
]
RSVG_CONVERT = '/home/wyw/miniconda3/envs/MSI/bin/rsvg-convert'


def command_rows(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False, text=True)
    except FileNotFoundError:
        return []
    return result.stdout.splitlines()


def pdf_image_count(path):
    lines = command_rows(['pdfimages', '-list', path])
    count = 0
    for line in lines:
        if re.match(r'^\s*\d+\s+\d+\s+', line):
            count += 1
    return count


def pdf_font_count(path):
    lines = command_rows(['pdffonts', path])
    count = 0
    for line in lines:
        if re.match(r'^\S+\s+', line) and not line.startswith(('name ', '---')):
            count += 1
    return count


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with ZipFile(DOCX, 'r') as archive:
        names = set(archive.namelist())
        for label, png_member, svg_member in FIGURES:
            if png_member not in names or svg_member not in names:
                raise RuntimeError('missing active manuscript media for %s' % label)
            png_path = os.path.join(OUT, label + '.png')
            svg_path = os.path.join(OUT, label + '.svg')
            pdf_path = os.path.join(OUT, label + '.pdf')
            with open(png_path, 'wb') as handle:
                handle.write(archive.read(png_member))
            svg_bytes = archive.read(svg_member)
            with open(svg_path, 'wb') as handle:
                handle.write(svg_bytes)

            # Figure 4 has a directly generated Matplotlib PDF. Other figures
            # are converted from the exact SVG embedded in the active DOCX.
            if label == 'Figure04_bulk':
                shutil.copy2(os.path.join(BRANCH, 'figures', 'figure4_bulk_reordered.pdf'), pdf_path)
            else:
                subprocess.run(
                    [RSVG_CONVERT, '-f', 'pdf', '-o', pdf_path, svg_path],
                    check=True,
                )

            lower_svg = svg_bytes.lower()
            svg_raster = (b'<image' in lower_svg or b'<foreignobject' in lower_svg)
            images = pdf_image_count(pdf_path)
            fonts = pdf_font_count(pdf_path)
            if not svg_raster and images == 0:
                assessment = 'vector SVG/PDF; editable paths and text objects'
            else:
                assessment = 'mixed vector/raster; inspect embedded image elements'
            rows.append({
                'figure': label,
                'svg': os.path.basename(svg_path),
                'pdf': os.path.basename(pdf_path),
                'png_preview': os.path.basename(png_path),
                'svg_contains_raster_image': svg_raster,
                'pdf_embedded_image_objects': images,
                'pdf_font_records': fonts,
                'illustrator_assessment': assessment,
            })

    for number in range(1, 4):
        label = 'SupplementaryFigure%02d' % number
        paths = {}
        for ext in ('svg', 'pdf', 'png'):
            source = os.path.join(SUPPLEMENT, label + '.' + ext)
            if not os.path.isfile(source):
                raise RuntimeError('missing supplementary figure: ' + source)
            paths[ext] = os.path.join(OUT, label + '.' + ext)
            shutil.copy2(source, paths[ext])
        with open(paths['svg'], 'rb') as handle:
            lower_svg = handle.read().lower()
        svg_raster = b'<image' in lower_svg or b'<foreignobject' in lower_svg
        images = pdf_image_count(paths['pdf'])
        fonts = pdf_font_count(paths['pdf'])
        if not svg_raster and images == 0:
            assessment = 'vector SVG/PDF; editable paths and text objects'
        else:
            assessment = 'mixed vector/raster; inspect embedded image elements'
        rows.append({
            'figure': label,
            'svg': os.path.basename(paths['svg']),
            'pdf': os.path.basename(paths['pdf']),
            'png_preview': os.path.basename(paths['png']),
            'svg_contains_raster_image': svg_raster,
            'pdf_embedded_image_objects': images,
            'pdf_font_records': fonts,
            'illustrator_assessment': assessment,
        })

    audit_path = os.path.join(OUT, 'editable_figure_audit.csv')
    with open(audit_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    readme = [
        '# Editable manuscript figures',
        '',
        'These files were exported from the media actually referenced by the active manuscript.',
        'Use SVG as the primary Adobe Illustrator editing source; matching PDF files are vector',
        'submission copies, and PNG files are raster previews only.',
        '',
        'Figure 2 is split into the main and continued panels because the manuscript embeds them',
        'as two independent figure objects.',
        'Supplementary Figures 1-3 are copied from the existing supplementary-figure directory.',
        'Fonts may require installation or substitution on another workstation to edit text.',
        '',
        '| Figure | SVG raster elements | PDF image objects | PDF font records | Assessment |',
        '|---|---:|---:|---:|---|',
    ]
    for row in rows:
        readme.append('| {figure} | {svg_contains_raster_image} | {pdf_embedded_image_objects} | {pdf_font_records} | {illustrator_assessment} |'.format(**row))
    with open(os.path.join(OUT, 'README.md'), 'w') as handle:
        handle.write('\n'.join(readme) + '\n')
    print('exported', len(rows), 'manuscript and supplementary figure objects to', OUT)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
