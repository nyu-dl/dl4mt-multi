#!/bin/bash
##############################################################################
# This script executes the preprocessing pipeline for the joint side of 
# multiway NMT models. 
# 
# In multi-way NMT, we occasionally share either some of the encoders, or some
# of the decoders, sometimes both. For example, in a multi-encoder NMT model, 
# decoder across all source-target pairs are shared, and in a multi-decoder
# model similarly, encoder across all pairs is shared. That's why, for the 
# shared encoders or decoders, we use a joint shared vocabulary, which is 
# build using a shared BPE encoding. In order to learn the joint BPE-codes,
# we first combine all the available data (eg. all the target side English
# data if we are building a multi-encoder) and then remove the duplicates.
# The combined (joint) data is then used to learn BPE-codes and all source or
# target side data is encoded by using this joint codes.
# 
# As we use Byte Pair Encoding for both source and target tokens, we need 
# subword-nmt repo available (For details about how to setup your environment
# for subword-nmt please check https://github.com/nyu-dl/dl4mt-tutorial).
# 
# We follow the steps below for single pair NMT pre-processing:
#   1. Combine all the source or target data
#   2. Remove duplicates in the combined corpus
#   3. Learn joint BPE-codes from the combined corpus
#   4. Encode training, development and test sets with the codes
#   5. Extract a joint dictionary from the encoded training set
#
# This script assumes the data associated with source and target side along 
# with development and test sets are downloaded, tokenized and put into the
# ${datadir}. Convenience scripts for downloading and tokenizing can be found
# again at dl4mt-tutorial.
##############################################################################

datadir1=${HOME}/data/bpe/de2en
datadir2=${HOME}/data/bpe/fi2en
datadir_joint=${HOME}/data/bpe/fide2en

source1_tr=${datadir1}/all.tok.apos.clean.shuf.de-en.en
source2_tr=${datadir2}/all.tok.apos.clean.shuf.fi-en.en

source1_dev=${datadir1}/newstest2013.en.tok.apos
source2_dev=${datadir2}/newsdev2015.en.tok.apos

source1_tst=${datadir1}/newstest2015-deen-ref.en.tok.apos
source2_tst=${datadir2}/newstest2015-fien-ref.en.tok.apos

learn_bpe=${HOME}/codes/subword-nmt/learn_bpe.py
apply_bpe=${HOME}/codes/subword-nmt/apply_bpe.py
dictionary=${HOME}/codes/GroundHog/experiments/nmt/preprocess/preprocess.py


##############################################################################
# Some utility functions and wrappers

# Encodes a given text file using code
encode () {
    code=$1
    inp=$2
    out=$3
    if [ -e "${out}" ]; then
        echo "${out} exists"
    else 
        echo "...encoding ${inp}"
        python ${apply_bpe} -c ${code} < ${inp} > ${out}
    fi 
}

# Extracts dictionary if it does not exist
extract_dict () {
    inp=$1
    out=$1.vocab.pkl
    if [ -e "${out}" ]; then
        echo "${out} exists"
    else 
        echo "...extracting dictionary ${inp}"
        python ${dictionary} ${inp} -d ${out}
    fi 
}

# Wrapper to call learn bpe
learn_bpe_call () {
    inp=$1
    out=$2
    n_sym=$3
    if [ -e "${out}" ]; then
        echo "${out} exists"
    else 
        echo "...learning bpe with ${n_sym} symbols using ${inp}"
        python ${learn_bpe} -s ${n_sym} < ${inp} > ${out}
    fi 
}

# Creates directory
create_dir () {
    if [ ! -d "${1}" ]; then
        mkdir -p ${1} 
    fi 
}

##############################################################################
# Prepare joint dataset and necessary folders etc 

create_dir ${datadir_joint}

# Merge two dataset first and remove duplicates
joint_data=${datadir_joint}/all.tok.apos.clean.fide-en.en
joint_data_nd=${joint_data}.noDup

echo "concatenating two files"
echo "file1 $(wc -l ${source1_tr})"
echo "file2 $(wc -l ${source2_tr})"
if [ ! -e "${joint_data}" ]; then
    cat ${source1_tr} ${source2_tr} >> ${joint_data}
fi

echo "joint file $(wc -l ${joint_data})"
echo "removing duplicates"
if [ ! -e "${joint_data_nd}" ]; then
    sort ${joint_data} | uniq >> ${joint_data_nd}
fi
echo "joint file no duplicates $(wc -l ${joint_data_nd})"


##############################################################################
# Learn BPEs and encode training, validation and test files for src and trg

echo "processing joint sides"
for n_symbols in 30000
do
    codes_file=${joint_data_nd}.code${n_symbols:0:2}k
    file_id=joint.bpe${n_symbols:0:2}k

    # Learn byte pair encoding given training data
    learn_bpe_call ${joint_data_nd} ${codes_file} ${n_symbols}

    # Encode training, development and test sets
    for inp in ${source1_tr} ${source1_dev} ${source1_tst} ${source2_tr} ${source2_dev} ${source2_tst} ${joint_data_nd}
    do
        encode ${codes_file} ${inp} ${inp}.${file_id}
    done

    echo; echo

done


##############################################################################
# Extract dictionaries using training data

echo "post-processing joint side"
for n_symbols in 30000
do
    file_id=joint.bpe${n_symbols:0:2}k

    # Extract dictionary using encoded training
    extract_dict ${joint_data_nd}.${file_id}

done
