#!/usr/bin/python
import sys


def avg_word_len(filename):
    word_lengths = []
    seq_lengths = []
    for line in open(filename).readlines():
        seq = [len(word) for word in line.split()]
        word_lengths.extend(seq)
        seq_lengths.append(len(seq))
    return float(sum(word_lengths))/float(len(word_lengths)), \
        float(sum(seq_lengths))/float(len(seq_lengths))


w_len, s_len = avg_word_len(sys.argv[1])
print 'word_len:', w_len, ' seq_len:', s_len
