import logging
import numpy
import os
import cPickle
import theano

from collections import OrderedDict
from itertools import chain

from blocks.roles import add_role, PARAMETER

from theano import tensor

rng = numpy.random.RandomState(4321)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('multiCG_utils')


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk, borrow=True)
        add_role(tparams[kk], PARAMETER)
    return tparams


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# revert back from prefix-appended name
def p_(name):
    return name.split('_')


# retrieve encoder ids from multiSrc cg name
def get_enc_ids(cg):
    if is_multiSource(cg):
        return p_(cg)[0].split('.')
    else:
        return p_(cg)[0]


# retrieve decoder ids from multiSrc cg name
def get_dec_ids(cg):
    return p_(cg)[1]


# extract encoder decoder ids from prefix-appended names
def get_enc_dec_ids(cgs):
    eids = set([p_(cgs[i])[0] for i in xrange(len(cgs))])
    dids = set([p_(cgs[i])[1] for i in xrange(len(cgs))])
    return sorted(eids), sorted(dids)


# extract sub dictionary given keys, preserves key order
def get_subdict(dict_, keys):
    return OrderedDict([(keys[i], dict_[keys[i]]) for i in range(len(keys))])


# some utilities
def ortho_weight(ndim):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype('float32')


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def seqs2words(caps, vocab, ivocab):
    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == vocab['</S>']:
                break
            ww.append(ivocab[w])
        capsw.append(' '.join(ww))
    return capsw


def words2seqs(seq, vocab, vocab_size, chr_level=False):
    if chr_level:
        words = list(seq.decode('utf-8').strip())
    else:
        words = seq.strip().split()
    x = map(
        lambda w: vocab[w] if w in vocab else
        vocab['<UNK>'], words)
    x = map(lambda ii: ii if ii < vocab_size else vocab['<UNK>'], x)
    x += [vocab['</S>']]
    return x


def words2seqs_multi(seqs, vocabs, vocab_sizes, chr_level=False):
    """All inputs must be a dictionary, except chr_level."""
    xs = OrderedDict()
    for sid in seqs.keys():
        if chr_level:
            words = list(seqs[sid].decode('utf-8').strip())
        else:
            words = seqs[sid].strip().split()
        x = map(
            lambda w: vocabs[sid][w] if w in vocabs[sid] else
            vocabs[sid]['<UNK>'], words)
        x = map(lambda ii: ii if ii < vocab_sizes[sid]
                else vocabs[sid]['<UNK>'], x)
        x += [vocabs[sid]['</S>']]
        xs[sid] = x
    return xs


def itemlist(tparams):
    '''get the list of parameters: Note that tparams must be OrderedDict'''
    return [vv for kk, vv in tparams.iteritems()]


def make_ordered_dict(item_list):
    '''make list of parameters ordered dictionary.'''
    return OrderedDict([(v.name, v) for v in item_list])


def extract_nbest_file(filename, n_best):
    '''Extracts nbest list into separate files.'''

    # create output files
    out_filenames = [filename + str(n) for n in range(n_best)]
    out_files = []
    for n in range(n_best):
        logger.info('Extracting into {}'.format(filename + str(n)))
        out_files.append(open(filename + str(n), 'w'))

    prev_idx = 0
    out_idx = -1
    with open(filename, 'r') as f:
        for line in f:
            idx, trans, score = line.split('|||')
            if prev_idx == int(idx):
                out_idx += 1
            else:
                prev_idx = int(idx)
                out_idx = 0
            print >> out_files[out_idx], trans

    # close output files
    for f in out_files:
        f.close()
    return out_filenames


def get_version(config):
    '''determines the version by looking at configurations.'''

    version = '0.0'
    if 'version' in config:
        return config['version']
    if 'dec_rnn_type' in config and \
            config['dec_rnn_type'].endswith('v08'):
        return '0.8'
    return version


class ReadOnlyDict(dict):
    '''Encapsulates a dictionary to make it readonly.'''
    def __setitem__(self, key, value):
        raise(TypeError, "__setitem__ is not supported")

    def __delitem__(self, key):
        raise(TypeError, "__delitem__ is not supported")

    def update(self, d):
        raise(TypeError, "update is not supported")


def get_paths(keys, ref_dict, basedir=''):
    '''Extracts keys given dictionary by basedir prepended.'''
    if not isinstance(keys, list):
        keys = [keys]
    ret = OrderedDict()
    for i in range(len(keys)):
        paths = ref_dict[keys[i]]
        if isinstance(paths, list):
            ret[keys[i]] = [os.path.join(basedir, paths[j])
                            for j in range(len(paths))]
        elif isinstance(paths, dict):
            ret[keys[i]] = {k: os.path.join(basedir, v)
                            for k, v in paths.items()}
        else:
            ret[keys[i]] = os.path.join(basedir, paths)
    return ret


def get_odict(keys, value=None):
    '''Preserves ordering in keys.'''
    return OrderedDict([(keys[i], value) for i in range(len(keys))])


def get_odict_pair(keys, values):
    '''Preserves ordering in keys, values as a list.'''
    return OrderedDict([(keys[i], values[i]) for i in range(len(keys))])


def get_val_set_outs(cgs, saveto):
    '''Formats validation set output file paths.'''
    return OrderedDict([
        (cgs[i], saveto + '/validation_out_{}2{}.txt'.format(*p_(cgs[i])))
        for i in range(len(cgs))])


def get_param_idx(plist, pname):
    '''Returns the index of a parameter given parameter name.'''
    idx = [i for i in range(len(plist)) if plist[i].name == pname]
    if len(idx) == 1:
        return idx[0]
    elif len(idx) > 1:
        raise ValueError('multiple items for parameter {} in plist {}'
                         .format(pname, plist))
    return None


def load_vocab(vocab_path, bos_idx=0, eos_idx=0, unk_idx=1):
    '''Loads a vocabulary from pickle file.'''
    vocab = cPickle.load(open(vocab_path))
    vocab['<S>'] = bos_idx
    vocab['</S>'] = eos_idx
    vocab['<UNK>'] = unk_idx
    return vocab


def invert_vocab(vocab):
    '''Inverts a dictionary keys and values.'''
    ivocab = dict()
    for kk, vv in vocab.iteritems():
        ivocab[vv] = kk
    return ivocab
