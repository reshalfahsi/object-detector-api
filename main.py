import argparse
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set important parameters', add_help=False)
    
    return parser


def main(args):

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)