"""Compare fresh paper outputs with immutable manuscript numerical records.

Timings and historical source/environment metadata are intentionally excluded.
All CSV scientific fields and every saved observation/weight/fit array are
checked. Missing files, changed rows, changed statuses, and nonfinite changes
fail verification. Small floating-point differences across BLAS builds are
allowed (rtol=2e-10, atol=2e-11); discrete fields are compared exactly.
"""
import csv
import json
import re
import numpy as np
from run_config import MODE, OUT, REFERENCE

RTOL, ATOL = 2e-10, 2e-11
TIMING = {'fit_seconds', 'elapsed_seconds', 'total_seconds', 'elapsed'}


def same(a, b, label):
    if isinstance(a, dict):
        for key in a:
            if key not in TIMING:
                assert key in b, f'{label}: missing {key}'
                same(a[key], b[key], f'{label}/{key}')
    elif isinstance(a, list):
        assert len(a) == len(b), f'{label}: changed length'
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{label}/{i}')
    elif isinstance(a, bool) or a is None:
        assert type(a) is type(b) and a == b, f'{label}: {a!r} != {b!r}'
    elif isinstance(a, int):
        assert type(b) is int and a == b, f'{label}: changed integer {a!r} != {b!r}'
    elif isinstance(a, float):
        assert np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True), f'{label}: {a!r} != {b!r}'
    elif isinstance(a, str) and re.fullmatch(r'[+-]?\d+', a):
        assert isinstance(b, str) and re.fullmatch(r'[+-]?\d+', b) and int(a) == int(b), f'{label}: changed integer {a!r} != {b!r}'
    else:
        try:
            x, y = float(a), float(b)
        except (ValueError, TypeError):
            assert a == b, f'{label}: {a!r} != {b!r}'
        else:
            assert np.isclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True), f'{label}: {a!r} != {b!r}'


def main():
    if MODE != 'paper':
        raise ValueError('Reference comparisons require paper mode')
    records, fields, arrays = [], 0, 0
    paths = sorted(p for p in REFERENCE.rglob('*') if p.suffix in ('.csv', '.npz') and p.name != 'problem_clouds.npz')
    for reference in paths:
        relative = reference.relative_to(REFERENCE)
        actual = OUT/relative
        assert actual.exists(), f'Missing paper output: {relative}'
        if reference.suffix == '.csv':
            with reference.open() as f, actual.open() as g:
                r, s = csv.DictReader(f), csv.DictReader(g)
                assert r.fieldnames == s.fieldnames, f'{relative}: changed CSV columns'
                left, right = list(r), list(s)
            same(left, right, str(relative))
            fields += len(left)*len(set(r.fieldnames)-TIMING)
            records.append(dict(file=str(relative), rows=len(left), passed=True))
        else:
            with np.load(reference, allow_pickle=False) as r, np.load(actual, allow_pickle=False) as s:
                assert set(r.files) <= set(s.files), f'{relative}: missing arrays'
                for key in r.files:
                    a, b = r[key], s[key]
                    assert a.shape == b.shape, f'{relative}/{key}: changed shape'
                    if a.dtype.kind in 'fc':
                        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f'{relative}/{key}')
                    else:
                        np.testing.assert_array_equal(a, b, err_msg=f'{relative}/{key}')
                    arrays += 1
                records.append(dict(file=str(relative), arrays=len(r.files), passed=True))
    for name in ('study_c_summary.json', 'study_c_failures.json', 'study_b/finite_tail_holm_summary.json'):
        same(json.loads((REFERENCE/name).read_text()), json.loads((OUT/name).read_text()), name)
        records.append(dict(file=name, passed=True))
    for name in ('study_a/numerical_manifest.json', 'study_b/numerical_manifest.json'):
        same(json.loads((REFERENCE/name).read_text()), json.loads((OUT/name).read_text()), name)
        records.append(dict(file=name, passed=True))
    report = dict(passed=True, mode=MODE, files_compared=len(records), csv_scientific_fields_compared=fields,
                  saved_arrays_compared=arrays, rtol=RTOL, atol=ATOL, excluded_fields=sorted(TIMING),
                  scope='Scientific CSV fields, retained arrays, Study C summaries/failures, Holm results, and A/B numerical diagnostics. Historical source hashes and timings excluded.',
                  files=records)
    (OUT/'reference_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
