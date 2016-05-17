"""
Plots field given multiple log files
"""
import argparse
import cPickle as pkl
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument("--log-files", nargs='+', type=str, help="Log files")
parser.add_argument("--item-names", nargs='+', type=str,
                    help="Names for legend files")
parser.add_argument("--cgs", help='Computational graphs to look into',
                    nargs='+')
parser.add_argument("--monitor", help='Variable to plot', type=str,
                    default='logprob')
parser.add_argument('--saveto', type=str, default="",
                    help="Print to file only")
parser.add_argument('--title', type=str, default="",
                    help="Generic title for each subplot")
parser.add_argument('--xlimit', type=int, default=None,
                    help="X-axis limit of plots")
args = parser.parse_args()

# append computational graphs to the field
items = [args.monitor + '_' + cg for cg in args.cgs]
num_models = len(args.log_files)
assert num_models == len(args.item_names), \
    "Item names must be equal to the number of log files"

# item names are used in legend
item_names = {k: k + ' of data' for k in args.item_names}

# x-limit is the maximum ticks in the x-axis of plot,
# used for truncating the logs that are too long
xlimit = (0 if args.xlimit is None else args.xlimit)

try:

    # iterate through log and fetch if field exists
    curves = OrderedDict()
    for i in range(num_models):
        d = {item: list() for item in items}
        log = pkl.load(open(args.log_files[i]))
        for _, dc in log.iteritems():
            inc = [x for x in items if x in dc]
            if len(inc):
                idxs = [items.index(ii) for ii in inc]
                for idx in idxs:
                    d[items[idx]].append(dc[inc[0]])
        curves[args.item_names[i]] = d

    # play with this to change the aspect ratio
    fig = plt.figure(figsize=(12, 10))
    for i, cg in enumerate(args.cgs):
        ax = plt.subplot(len(args.cgs), 1, i+1)
        for ii, (curve_name, curve_dict) in enumerate(curves.items()):
            vals_to_plot = curves[curve_name][args.monitor + '_' + cg]
            vals_to_plot = (vals_to_plot[:xlimit]
                            if xlimit > 0 else vals_to_plot)
            plt.plot(
                np.asarray(range(len(vals_to_plot))),
                np.asarray(vals_to_plot),
                label=item_names[curve_name])
        plt.legend(loc='lower right', shadow=True)

        # logprob is computed once in every 2k updates
        plt.xlabel('Iter (x2000)')
        plt.ylabel('NLL')
        plt.title(args.title + cg)

    plt.tight_layout()
    if args.saveto != "":
        plt.savefig(args.saveto)
    else:
        plt.show()

except Exception as e:
    print str(e)
