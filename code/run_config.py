"""Shared run selection for scripts and notebooks; frozen references are read-only."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get('GID_MODE', 'smoke')
if MODE not in ('smoke', 'paper'):
    raise ValueError('GID_MODE must be smoke or paper')
OUT = Path(os.environ.get('GID_OUTPUT_ROOT', str(ROOT / 'results' / MODE))).expanduser().resolve()
REFERENCE = ROOT / 'reference_results'
PROTOCOLS = ROOT / 'protocols'
PLOTS = os.environ.get('GID_PLOTS', '0') == '1'
if OUT == REFERENCE or REFERENCE in OUT.parents:
    raise ValueError('Generated output must not overwrite reference_results')
OUT.mkdir(parents=True, exist_ok=True)


def repetitions(paper_count):
    return int(paper_count) if MODE == 'paper' else 5
