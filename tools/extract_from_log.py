"""
Extracts fields from log into file
"""
import argparse
import numpy
import cPickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument("--log-file", type=str, help="Log file")
parser.add_argument("--item", help='Variable to extract', type=str,
                    default='logprob')
parser.add_argument('--saveto', type=str, default="",
                    help="Numpy file to save")
parser.add_argument("--cgs", help='Computational graphs to look into',
                    nargs='+')
args = parser.parse_args()


# append computational graphs to the field
items = [args.item + '_' + cg for cg in args.cgs]

# allocate output data and load log
data = {item: list() for item in items}
log = pkl.load(open(args.log_file))

# iterate through log and fetch if field exists
for _, dc in log.iteritems():
    inc = [x for x in items if x in dc]
    if len(inc):
        idxs = [items.index(ii) for ii in inc]
        for idx in idxs:
            data[items[idx]].append(dc[inc[idx]])

# save extracted data into npz file
print 'saving to [{}]'.format(args.saveto)
numpy.savez(args.saveto, **data)
