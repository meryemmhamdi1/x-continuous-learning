import json

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

# Global statistics that we can output to monitor the run.


stats_path = None
STATS = {}


def init(path):
    global stats_path, STATS
    stats_path = path
    try:
        STATS = read_json(stats_path)
    except Exception:
        STATS = {}


def add(*args):
    # Example: add_stats('data', 'num_examples', 3)
    s = STATS
    prefix = args[:-2]
    for k in prefix:
        if k not in s:
            s[k] = {}
        s = s[k]
    s[args[-2]] = args[-1]
    flush()


def add_args(key, args):
    add(key, dict((arg, getattr(args, arg)) for arg in vars(args)))


def flush():
    if stats_path:
        out = open(stats_path, 'w')
        print(json.dumps(STATS))
        out.close()


def write_json(raw, path):
    with open(path, 'w') as out:
        json.dump(raw, out, indent=4, sort_keys=True)

