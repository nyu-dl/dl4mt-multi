"""
Reverts back from merged file.
Adapted from Sebastien Jean's version.
"""
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cname", type=str)
parser.add_argument("lname", type=str)
parser.add_argument("rname", type=str)
args = parser.parse_args()

with open(args.cname) as combined:
    with open(args.lname, 'w') as left:
        with open(args.rname, 'w') as right:
            for line in combined:
                line = line.split('|||')
                left.write(line[0].strip() + '\n')
                right.write(line[1].strip() + '\n')
