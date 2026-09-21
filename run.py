"""Run manuscript experiments: python run.py --mode paper --figures --verify."""
import argparse
from workflow import run_study, verify_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'paper'], default='smoke')
    parser.add_argument('--study', choices=['all', 'a', 'b', 'c', 'checks'], default='all')
    parser.add_argument('--figures', action='store_true', help='Save PGF/PDF/PNG figures using pdflatex')
    parser.add_argument('--verify', action='store_true', help='Compare a complete paper run with frozen manuscript outputs')
    parser.add_argument('--output-root', help='Optional separate generated-output directory')
    args = parser.parse_args()
    if args.verify and (args.mode != 'paper' or args.study != 'all'):
        parser.error('--verify requires --mode paper --study all')
    out = run_study(args.study, args.mode, args.figures, args.output_root)
    if args.verify:
        verify_paper(out)
    print(f'Completed {args.mode} run. Results: {out}')


if __name__ == '__main__':
    main()
