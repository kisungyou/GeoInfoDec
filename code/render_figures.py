"""Render every Matplotlib figure from stored results, without refitting.

Run through python run.py --mode paper --figures for a full reproduction
Requires pdflatex and a PDF-to-PNG converter supported by Matplotlib PGF.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime,timezone
from figure_style import matplotlib,RCPARAMS,FONT_SIZE_PT,TEXT_WIDTH_IN
from make_problem_figure import main as problem
from study_ab import plot_a,plot_b
from study_c_digits import plot as heldout

from run_config import ROOT, OUT, MODE


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return list(csv.DictReader(path.open()))
def scientific_inputs():
    return {str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.suffix in ('.csv','.json','.npz') and p.name != 'figure_verification.json'}

def command_output(command):
    if shutil.which(command[0]) is None:return None
    return subprocess.run(command,capture_output=True,text=True,check=True).stdout


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=scientific_inputs()
    result=[]
    result.append(problem())
    a=OUT/'study_a'
    replicate_rows=[r for path in sorted(a.glob('*_replicates.csv')) for r in rows(path)]
    result.append(plot_a(a,rows(a/'rejection_summary.csv'),replicate_rows))
    result.append(heldout(rows(OUT/'study_c_queries.csv')))
    b=OUT/'study_b'
    result.append(plot_b(b,rows(b/'power_estimation_summary.csv')))
    for item in result:
        pgf,pdf,png=map(Path,item['files'])
        text=pgf.read_text()
        fonts=sorted({float(x) for x in re.findall(r'\\fontsize\{([\d.]+)\}',text)})
        assets=sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)))
        info=command_output(['pdfinfo',str(pdf)])
        fontinfo=command_output(['pdffonts',str(pdf)])
        dims=re.search(r'Page size:\s*([\d.]+) x ([\d.]+) pts',info or '')
        canvas=[float(dims.group(i)) for i in (1,2)] if dims else None
        item.update(files=[str(p.relative_to(OUT)) for p in (pgf,pdf,png)],
            pgf_text_font_sizes_tex_pt=fonts,pgf_external_graphic_assets=assets,
            pdf_canvas_big_points=canvas,pdf_fonts=fontinfo,
            file_sha256={str(p.relative_to(OUT)):sha(p) for p in (pgf,pdf,png)})
        assert fonts==[float(FONT_SIZE_PT)],(pgf,fonts)
        assert abs(item['width_inches']-TEXT_WIDTH_IN)<1e-12
        if canvas:
            assert abs(canvas[0]-72*TEXT_WIDTH_IN)<.02
            assert abs(canvas[1]-72*item['height_inches'])<.02
        for asset in assets:assert (pgf.parent/asset).exists(),asset
    after=scientific_inputs()
    assert before==after,'Rendering altered retained scientific inputs'
    report=dict(mode=MODE,created_utc=datetime.now(timezone.utc).isoformat(),
        font='Computer Modern Roman, cmr10',font_size_tex_pt=FONT_SIZE_PT,
        text_width_inches=TEXT_WIDTH_IN,backend=matplotlib.get_backend(),
        texsystem=matplotlib.rcParams['pgf.texsystem'],
        font_evidence='Actual aistats2027.sty compiled in font-probe.tex: cmr10; 10 TeX pt; textwidth487.8225 TeX pt =6.75in.',
        inclusion='Natural-size PDF or unscaled PGF; no width=, resizebox, or scalebox.',
        plot_rcparams={k:(str(v) if k=='axes.prop_cycle' else v) for k,v in RCPARAMS.items()},
        figures=result,scientific_inputs_unchanged=True,
        scientific_input_file_count=len(before),scientific_input_sha256=before,
        exceptions='No reduced-size prose, ticks, legends, or titles. Mathematical subscripts retain ordinary TeX script sizes, as in the manuscript.',
        sidecar_policy='All PGF images are vector; the generated PGF files require no raster sidecars.',
        rendering_scope='Stored simulation/query outputs and illustrative clouds only; no simulations, model fits, estimates, or statistical tests rerun.')
    (OUT/'figure_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('font','font_size_tex_pt','backend','scientific_inputs_unchanged','scientific_input_file_count')},indent=2))
    for item in result:print(item['files'][1],item['width_inches'],item['height_inches'],item['pgf_text_font_sizes_tex_pt'])

if __name__=='__main__':main()
