#!/bin/bash
##############################################################################
# This script shuffles bitext n-times and appends each shuffled file into 
# final large files for source and target. We use this script since our data
# iterator does not shuffle bitext after each epoch.
# 
# usage: shuf_parallel.py src_in trg_in src_out trg_out ntimes
#
##############################################################################

src=${HOME}/git/ubiqml/mlnmt/scripts
merge_script=${src}/format_fast_align.py
unmerge_script=${src}/unmerge.py

in1=$1
in2=$2

out1=$3
out2=$4

ntimes=$5

# set random seed first
seed=4444
RANDOM=$seed

combined=./combined.tmp
shuf_combined=./combined.shuf.tmp

echo "merging..."
python ${merge_script} ${in1} ${in2} ${combined}


for i in `seq 1 ${ntimes}`; do
    echo "shuffling ${i}..."
    shuf ${combined} >> ${shuf_combined}
done

echo "unmerging..."
python ${unmerge_script} ${shuf_combined} ${out1} ${out2}

rm ${combined}
rm ${shuf_combined}
