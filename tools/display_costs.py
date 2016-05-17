import argparse
import cPickle as pkl
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from multiCG_utils import _p

parser = argparse.ArgumentParser()
parser.add_argument("--log-file", type=str, help="Log file")
parser.add_argument("--enc-list", help='Encoder language names input',
                    nargs='+')
parser.add_argument("--dec-list", help='delimited list input', nargs='+')
parser.add_argument("--monitor", help='Variable to plot', type=str,
                    default='cost')
parser.add_argument('--saveto', type=str, default="",
                    help="Print to file only")
args = parser.parse_args()

# Change this if necessary
items = []
for enc, dec in itertools.product(args.enc_list, args.dec_list):
    items.append(args.monitor + '_' + _p(enc, dec))

try:
    d = {}
    log = pkl.load(open(args.log_file))
    for _, dc in log.iteritems():
        inc = [x for x in items if x in dc]
        if len(inc):
            assert len(inc) == 1
            idx = items.index(inc[0])
            if items[idx] not in d:
                d[items[idx]] = list()
            d[items[idx]].append(dc[inc[0]])

    fig = plt.figure(figsize=(12, 10))
    for i, (key, val) in enumerate(d.iteritems()):
        plt.subplot(len(d), 1, i)
        plt.plot(np.asarray(xrange(len(val))), np.asarray(val))
        plt.xlabel('Iter')
        plt.ylabel('Cost')
        plt.title(key)

    plt.tight_layout()
    if args.saveto != "":
        plt.savefig(args.saveto)
    else:
        plt.show()

except Exception as e:
    print str(e)

