"""Shared entry point used by the command line and all four notebooks."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
STUDIES = {
    'a': [('study_ab.py', 'a')],
    'b': [('study_ab.py', 'b'), ('study_b_supplement.py',)],
    'c': [('study_c_digits.py',), ('study_c_validate.py',)],
    'checks': [('numerical_checks.py',), ('refinement_gate_checks.py',)],
}


def run_study(study='all', mode='smoke', figures=False, output_root=None):
    """Execute fresh fits; return the directory holding this mode's results.

    Smoke reduces repetition counts only. Paper uses the frozen manuscript
    protocol. Saved reference results are never used as fitted run outputs.
    """
    if mode not in ('smoke', 'paper'):
        raise ValueError('mode must be smoke or paper')
    if study not in (*STUDIES, 'all'):
        raise ValueError('study must be all, a, b, c, or checks')
    if figures and shutil.which('pdflatex') is None:
        raise RuntimeError('Figure generation requires pdflatex on PATH; run without figures for numerical results.')
    out = (Path(output_root).expanduser() if output_root else ROOT / 'results' / mode).resolve()
    protected = [ROOT / name for name in ('code', 'notebooks', 'protocols', 'reference_results', 'reference_figures', 'checks', 'licenses', '.git')]
    if out == ROOT or any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('Output directory must be separate from source and frozen references')
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'run_mode.json'
    existing = [marker, *out.glob('run_*.json'), out/'study_a'/'config.json',
                out/'study_b'/'config.json', out/'study_c_metadata.json']
    for record_path in existing:
        if record_path.exists():
            recorded_mode = json.loads(record_path.read_text()).get('mode')
            if recorded_mode is not None and recorded_mode != mode:
                raise ValueError(f'Output directory contains {recorded_mode} results; choose a separate directory for {mode}')
    marker.write_text(json.dumps({'mode': mode}, indent=2)+'\n')
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(GID_MODE=mode, GID_OUTPUT_ROOT=str(out),
               GID_PLOTS='1' if figures and study != 'all' else '0',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'gid-pgf-matplotlib'))
    commands = [cmd for s in ('a', 'b', 'c', 'checks') for cmd in STUDIES[s]] if study == 'all' else list(STUDIES[study])
    if study == 'all':
        commands.append(('summarize_ab.py',))
        if figures:
            commands.append(('render_figures.py',))
    runs = []
    for command in commands:
        label = '-'.join(command).replace('.py', '')
        print(f'{mode}: {label}', flush=True)
        started = time.perf_counter()
        with (logs / (label + '.log')).open('w') as log:
            process = subprocess.run([sys.executable, str(ROOT / 'code' / command[0]), *command[1:]],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(command=['python', 'code/' + command[0], *command[1:]],
                      returncode=process.returncode, elapsed_seconds=time.perf_counter()-started)
        runs.append(record)
        if process.returncode:
            tail = '\n'.join((logs / (label + '.log')).read_text().splitlines()[-25:])
            raise RuntimeError(f'{label} failed:\n{tail}')
    sources = [ROOT/'workflow.py', ROOT/'run.py', *sorted((ROOT/'code').glob('*.py'))]
    record = dict(mode=mode, study=study, figures=figures, runs=runs,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  purpose='Execution check only; too few repetitions for manuscript claims.' if mode == 'smoke' else 'Frozen manuscript-scale protocol.')
    (out / ('run_' + study + '.json')).write_text(json.dumps(record, indent=2)+'\n')
    return out


def verify_paper(output_root=None):
    """Compare regenerated paper-scale scientific outputs to frozen records."""
    out = (Path(output_root) if output_root else ROOT/'results'/'paper').resolve()
    env = dict(os.environ, GID_MODE='paper', GID_OUTPUT_ROOT=str(out), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(ROOT/'code'/'compare_reference.py')], cwd=ROOT, env=env, check=True)
    return json.loads((out/'reference_comparison.json').read_text())
