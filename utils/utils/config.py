import argparse
import sys
import json
import os


def get_parser(args):
    parser = argparse.ArgumentParser()
    for arg, value in args.items():
        if isinstance(value, dict):
            parser.add_argument('--%s' % arg, type=json.loads, default=value)
        else:
            parser.add_argument('--%s' % arg, type=type(value), default=value)
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    for dir in [args.output_dir, args.model_dir, args.log_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return args