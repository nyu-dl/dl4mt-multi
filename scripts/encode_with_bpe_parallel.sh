#!/bin/bash
##############################################################################
# This script executes the preprocessing pipeline for a single pair NMT model. 
# As we use Byte Pair Encoding for both source and target tokens, we need 
# subword-nmt repo available (For details about how to setup your environment
# for subword-nmt please check https://github.com/nyu-dl/dl4mt-tutorial).
# 
# We follow the steps below for single pair NMT pre-processing:
#   1. Learn BPE-codes for both source and target side
#   2. Encode training, development and test sets with the codes
#
# This script assumes the data associated with source and target side along 
# with development and test sets are downloaded and put into the ${datadir}.

# Convenience scripts for downloading can be found again at dl4mt-tutorial.
##############################################################################

datadir=${HOME}/data/bpe/fi2en

source_tr=${datadir}/all.tok.apos.clean.shuf.fi-en.fi
target_tr=${datadir}/all.tok.apos.clean.shuf.fi-en.en

source_dev=${datadir}/newsdev2015.fi.tok.apos
target_dev=${datadir}/newsdev2015.en.tok.apos

source_tst=${datadir}/newstest2015-fien-src.fi.tok.apos
target_tst=${datadir}/newstest2015-fien-ref.en.tok.apos

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


##############################################################################
# Learn BPEs and encode training, validation and test files for src and trg

echo "processing source side"
for n_symbols in 30000
do
    codes_file=${source_tr}.code${n_symbols:0:2}
    file_id=bpe${n_symbols:0:2}k

    # Learn byte pair encoding given training data
    learn_bpe_call ${source_tr} ${codes_file} ${n_symbols}

    # Encode training, development and test sets
    for inp in ${source_tr} ${source_dev} ${source_tst}
    do
        encode ${codes_file} ${inp} ${inp}.${file_id}
    done
    echo; echo
done

echo "processing target side"
for n_symbols in 30000
do
    codes_file=${target_tr}.code${n_symbols:0:2}
    file_id=bpe${n_symbols:0:2}k

    # Learn byte pair encoding given training data
    learn_bpe_call ${target_tr} ${codes_file} ${n_symbols}

    # Encode training, development and test sets
    for inp in ${target_tr} ${target_dev} ${target_tst}
    do
        encode ${codes_file} ${inp} ${inp}.${file_id}
    done
    echo; echo
done


##############################################################################
# Extract dictionaries using training data 

echo "post-processing source side"
for n_symbols in 10000 30000 50000 80000
do
    file_id=bpe${n_symbols:0:2}k

    # Extract dictionary using encoded training
    extract_dict ${source_tr}.${file_id}

done

echo "post-processing target side"
for n_symbols in 10000 30000 50000 80000
do
    file_id=bpe${n_symbols:0:2}k

    # Extract dictionary using encoded training
    extract_dict ${target_tr}.${file_id}

done
