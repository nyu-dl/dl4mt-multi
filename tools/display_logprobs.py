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
parser.add_argument("--cg-names", help='CG names', nargs='+')
parser.add_argument("--monitor", help='Variable to plot', type=str,
                    default='logprob')
parser.add_argument('--saveto', type=str, default="",
                    help="Print to file only")
args = parser.parse_args()

# Change this if necessary
items = []
if args.cg_names is not None:
    items = [_p(args.monitor, cg) for cg in args.cg_names]
else:
    for enc, dec in itertools.product(args.enc_list, args.dec_list):
        if enc != dec:
            items.append(args.monitor + '_' + _p(enc, dec))

d = {}
log = pkl.load(open(args.log_file))
for _, dc in log.iteritems():
    inc = [x for x in items if x in dc]
    if len(inc):
        idxs = [items.index(ii) for ii in inc]
        for idx in idxs:
            if items[idx] not in d:
                d[items[idx]] = list()
            d[items[idx]].append(dc[inc[idx]])

fig = plt.figure(figsize=(12, 10))
for i, (key, val) in enumerate(d.iteritems()):
    plt.subplot(len(d), 1, i)
    plt.plot(np.asarray(xrange(len(val))), np.asarray(val))
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title(key)
    print key
    print np.asarray(val)

plt.tight_layout()
if args.saveto != "":
    plt.savefig(args.saveto)
else:
    plt.show()
