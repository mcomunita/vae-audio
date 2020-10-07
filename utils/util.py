import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def print_json(content):
    print(json.dumps(content, indent=4, sort_keys=False))

def get_instance(module, name, config):
    func_args = config[name]['args'] if 'args' in config[name] else None
    # if any argument specified in config[name]['args']
    if func_args:
        return getattr(module, config[name]['type'])(**func_args)
    # if not then just return the module
    return getattr(module, config[name]['type'])()

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
