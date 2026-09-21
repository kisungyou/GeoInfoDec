"""Regression tests for mode isolation and exact discrete reference checks."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow import run_study


class WorkflowChecks(unittest.TestCase):
    def test_cross_mode_output_is_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            marker = out/'run_mode.json'
            marker.write_text(json.dumps({'mode': 'paper'}))
            sentinel = out/'result.csv'
            sentinel.write_text('retained paper result\n')
            with self.assertRaisesRegex(ValueError, 'contains paper results'):
                run_study('checks', mode='smoke', output_root=out)
            self.assertEqual(sentinel.read_text(), 'retained paper result\n')
            self.assertEqual(json.loads(marker.read_text())['mode'], 'paper')
            self.assertFalse((out/'logs').exists())

    def test_frozen_reference_directory_is_protected(self):
        with self.assertRaisesRegex(ValueError, 'separate from source'):
            run_study('checks', output_root=ROOT/'reference_results')

    def test_integer_and_status_mutations_fail_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            prior = os.environ.get('GID_OUTPUT_ROOT')
            os.environ['GID_OUTPUT_ROOT'] = folder
            sys.path.insert(0, str(ROOT/'code'))
            try:
                spec = importlib.util.spec_from_file_location('gid_compare_test', ROOT/'code'/'compare_reference.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for a, b in [(1, 1.00000000001), ('2026091601', '2026091601.1'), ('ok', 'failed'), (True, False), (True, 1)]:
                    with self.subTest(original=a, changed=b), self.assertRaises(AssertionError):
                        module.same(a, b, 'changed discrete field')
                module.same(0.7, 0.7+1e-12, 'allowed floating-point rounding')
            finally:
                sys.path.pop(0)
                if prior is None:
                    os.environ.pop('GID_OUTPUT_ROOT', None)
                else:
                    os.environ['GID_OUTPUT_ROOT'] = prior


if __name__ == '__main__':
    unittest.main()
