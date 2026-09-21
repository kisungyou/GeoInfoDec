"""Shared manuscript figure style: PGF/pdflatex, Computer Modern, 10 TeX pt.

The actual AISTATS2027 document was probed: its main text uses cmr10 at
10 TeX pt and text width 6.75 inches. PDF and PGF are saved on an identical
fixed canvas. Do not rescale them when including them in the manuscript.
"""
from pathlib import Path
import io
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gid-pgf-matplotlib'))
import matplotlib
matplotlib.use('pgf')
from cycler import cycler

TEXT_WIDTH_IN = 6.75
FONT_SIZE_PT = 10
COLORS = dict(blue='#0072B2', orange='#D55E00', purple='#8E4779',
              gray='#757575', black='#242424')
RCPARAMS = {
    'pgf.texsystem': 'pdflatex',
    'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage{amsmath,amssymb}',
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': FONT_SIZE_PT,
    'axes.labelsize': FONT_SIZE_PT,
    'axes.titlesize': FONT_SIZE_PT,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'xtick.labelsize': FONT_SIZE_PT,
    'ytick.labelsize': FONT_SIZE_PT,
    'legend.fontsize': FONT_SIZE_PT,
    'legend.title_fontsize': FONT_SIZE_PT,
    'figure.titlesize': FONT_SIZE_PT,
    'figure.labelsize': FONT_SIZE_PT,
    'axes.prop_cycle': cycler(color=[COLORS['blue'],COLORS['orange'],COLORS['purple']]),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': .65,
    'axes.edgecolor': COLORS['black'],
    'axes.labelcolor': COLORS['black'],
    'text.color': COLORS['black'],
    'axes.titlepad': 6,
    'axes.labelpad': 4,
    'axes.unicode_minus': False,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': .65,
    'ytick.major.width': .65,
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
    'xtick.color': COLORS['black'],
    'ytick.color': COLORS['black'],
    'lines.linewidth': 1.15,
    'lines.markersize': 4,
    'patch.linewidth': .65,
    'legend.frameon': False,
    'legend.borderaxespad': .3,
    'legend.handlelength': 1.7,
    'legend.handletextpad': .5,
    'legend.columnspacing': 1.3,
    'savefig.bbox': None,
    'savefig.pad_inches': 0,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
}
matplotlib.rcParams.update(RCPARAMS)
import matplotlib.pyplot as plt


def save_figure(fig, stem, png=True):
    """Save paired PGF/PDF at natural size; PNG is a review convenience."""
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    for fmt in ('pgf','pdf'):
        fig.savefig(stem.with_suffix('.'+fmt),format=fmt,bbox_inches=None)
    if png:
        fig.savefig(stem.with_suffix('.png'),format='png',dpi=180,bbox_inches=None)
    # Recompute positions at the restored native DPI after the PNG save.
    # Otherwise cached legend positions still reflect the PNG's higher DPI.
    from matplotlib.backends.backend_pgf import RendererPgf
    renderer=RendererPgf(fig,io.StringIO())
    fig.draw(renderer)
    canvas=fig.bbox
    text_records=[]
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        outside=(box.x0<canvas.x0-.5 or box.y0<canvas.y0-.5 or
                 box.x1>canvas.x1+.5 or box.y1>canvas.y1+.5)
        text_records.append(dict(text=artist.get_text(),fontsize=float(artist.get_fontsize()),
            bounds_pixels=[float(x) for x in (box.x0,box.y0,box.x1,box.y1)],outside_canvas=bool(outside)))
    clipped=[x for x in text_records if x['outside_canvas']]
    if clipped:raise RuntimeError(f'Text outside figure canvas: {clipped}')
    return dict(text_bounding_boxes=text_records,text_clipping_check_passed=True,width_inches=float(fig.get_figwidth()),height_inches=float(fig.get_figheight()),
                font_size_tex_pt=FONT_SIZE_PT,backend=matplotlib.get_backend(),
                files=[str(stem.with_suffix('.'+fmt)) for fmt in ('pgf','pdf','png') if png or fmt!='png'])
