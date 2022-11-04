import argparse
import sys
import json


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
    return args

def get_conf_from_json(conf_file_path, param_map=None):
    _param_map = {}

    with open(conf_file_path, "r") as f:
        config = json.load(f)
    if not config:
        raise "config file not exists"
    if config['parameters']:
        _param_map = config['parameters']
    if param_map:
        _param_map.update(param_map)
    return _param_map