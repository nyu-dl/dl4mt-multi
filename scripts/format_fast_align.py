"""
Convert a tokenized parallel corpus into a format suitable for fast_align
Adapted from Sebastien Jean's version.
"""
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lname", type=str) # Use the tokenized text files # Source
parser.add_argument("rname", type=str) # Target
parser.add_argument("finalname", type=str)
args = parser.parse_args()

with open(args.lname, 'r') as left:
    with open(args.rname, 'r') as right:
        with open(args.finalname, 'w') as final:
            while True:
                lline = left.readline()
                rline = right.readline()
                if (lline == '') or (rline == ''):
                    break
                assert (lline[-1] == '\n')
                assert (rline[-1] == '\n')
                if (lline != '\n') and (rline != '\n'):
                    final.write(lline[:-1] + ' ||| ' + rline)
