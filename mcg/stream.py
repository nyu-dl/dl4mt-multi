import cPickle
import logging
import numpy
import six
import theano

from collections import OrderedDict

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Transformer, Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from .utils import p_, get_enc_dec_ids, get_num_lines

logger = logging.getLogger(__name__)


class MultiEncStream(Transformer, six.Iterator):
    """Stream manager that selects among multiple streams."""

    def __init__(self, streams, schedule, batch_sizes, transpose=False,
                 start_after=None):

        self.streams = streams
        self.schedule = OrderedDict(schedule)
        self.cgs = schedule.keys()
        self.curr_epoch_iterator = None
        self.batch_sizes = batch_sizes
        self.transpose = transpose
        self.start_after = start_after

        # Some streams may start after some iterations
        if self.start_after is None:
            self.start_after = OrderedDict([(k, 0) for k in self.cgs])

        # Select cg to start
        self.curr_id = [k for k, v in self.start_after.items() if v == 0][0]

        # Counters
        self.enc_ids, self.dec_ids = get_enc_dec_ids(self.cgs)
        self.counters = OrderedDict(
            [(cg, 0) for cg in schedule.keys()])
        self.training_counter = OrderedDict(
            [(cg, 0) for cg in schedule.keys()])
        self.epoch_counter = OrderedDict(
            [(cg, 0) for cg in schedule.keys()])
        self.num_encs = len(self.enc_ids)
        self.num_decs = len(self.dec_ids)

        # Get all epoch iterators
        self.epoch_iterators = OrderedDict(
            [(k, v.get_epoch_iterator(as_dict=True))
             for k, v in self.streams.iteritems()])

        # Initialize epoch iterator id to zero
        self.curr_epoch_iterator = self.epoch_iterators[self.curr_id]

        # Helps to initialize the random schedule
        self.fix_schedule_after_reload(self.schedule)

    def fix_schedule_after_reload(self, schedule):
        self.schedule = schedule
        self.random_stream = False
        # if all iterations are -1 we have random schedule
        if numpy.all([ii == -1 for ii in schedule.values()]):
            self.r_stream_id = numpy.random.randint(0, len(self.cgs), 2000000)
            self.random_stream = True

    def get_epoch_iterator(self, **kwargs):
        return self

    def __next__(self):
        batch = self._get_batch_with_reset(
            self.epoch_iterators[self.curr_id])
        src_id, trg_id = p_(self.curr_id)
        self._add_selectors(batch,
                            list(self.enc_ids).index(src_id),
                            list(self.dec_ids).index(trg_id))
        self._update_counters()
        return batch

    def _add_selectors(self, batch, src_id, trg_id):
        """Set src and target selector vectors"""
        batch_size = batch['source'].shape[1]
        batch['src_selector'] = numpy.zeros(
            (batch_size, self.num_encs)).astype(theano.config.floatX)
        batch['src_selector'][:, src_id] = 1.
        batch['trg_selector'] = numpy.zeros(
            (batch_size, self.num_decs)).astype(theano.config.floatX)
        batch['trg_selector'][:, trg_id] = 1.

    def _update_counters(self):

        # Increment counter and check schedule
        self.training_counter[self.curr_id] += 1
        self.counters[self.curr_id] += 1

        # Change stream according to schedule
        if self.random_stream:
            num_iter = sum(self.training_counter.values())
            self.curr_id = self.cgs[self.r_stream_id[num_iter]]
        elif self.counters[self.curr_id] // self.schedule[self.curr_id]:
            num_iter = sum(self.training_counter.values())
            self.counters[self.curr_id] = 0
            self.curr_id = self._get_next_stream_id(self.curr_id, num_iter,
                                                    len(self.streams))
            self.curr_epoch_iterator = self.epoch_iterators[self.curr_id]

    def _get_next_stream_id(self, curr_id, num_iter, num_streams):
        next_id = self.cgs[(self.cgs.index(curr_id) + 1) % num_streams]
        if not hasattr(self, 'start_after'):
            return next_id
        if self.start_after[next_id] > num_iter:
            return self._get_next_stream_id(next_id, num_iter, num_streams)
        return next_id

    def get_batch_with_stream_id(self, stream_id):
        batch = self._get_batch_with_reset(self.epoch_iterators[stream_id])
        src_id, trg_id = p_(stream_id)
        self._add_selectors(batch,
                            self.enc_ids.index(src_id),
                            self.dec_ids.index(trg_id))
        return batch

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_batch_with_reset(self, epoch_iterator):
        while True:
            try:
                batch = next(epoch_iterator)
                if self.transpose:
                    for k, v in batch.iteritems():
                        batch[k] = v.T
                return batch
            # TODO: This may not be the only source of exception
            except:
                sources = self._get_attr_rec(
                    epoch_iterator, 'data_stream').data_streams
                # Reset streams
                for st in sources:
                    st.reset()
                # Increment epoch counter
                self._update_epoch_counter(epoch_iterator)

    def _update_epoch_counter(self, epoch_iterator):
        idx = [k for (k, t) in self.epoch_iterators.iteritems()
               if t == epoch_iterator][0]
        self.epoch_counter[idx] += 1


def _length(sentence_pair):
    '''Assumes target is the last element in the tuple'''
    return len(sentence_pair[-1])


class _remapWordIdx(object):
    def __init__(self, mappings):
        self.mappings = mappings

    def __call__(self, sentence_pair):
        for mapping in self.mappings:
            if mapping[2] == mapping[1]:
                continue
            sentence_pair[mapping[0]][numpy.where(
                sentence_pair[mapping[0]] == mapping[1])] = mapping[2]
        return sentence_pair


class _oov_to_unk(object):
    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair, dummy):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])


class _too_short(object):
    def __init__(self, seq_len=10):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) >= self.seq_len
                    for sentence in sentence_pair])


def get_tr_stream(config):

    cgs = config['cgs']
    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Prepare source vocabs and files, make sure special tokens are there
    src_files = config['src_datas']
    src_vocabs = {k: cPickle.load(open(v))
                  for k, v in config['src_vocabs'].iteritems()}
    for k in src_vocabs.keys():
        src_vocabs[k]['<S>'] = 0
        src_vocabs[k]['</S>'] = config['src_eos_idxs'][k]
        src_vocabs[k]['<UNK>'] = config['unk_id']

    # Prepare target vocabs and files, make sure special tokens are there
    trg_files = config['trg_datas']
    trg_vocabs = {k: cPickle.load(open(v))
                  for k, v in config['trg_vocabs'].iteritems()}
    for k in trg_vocabs.keys():
        trg_vocabs[k]['<S>'] = 0
        trg_vocabs[k]['</S>'] = config['trg_eos_idxs'][k]
        trg_vocabs[k]['<UNK>'] = config['unk_id']

    # Create individual source streams
    src_datasets = {
        cg: TextFile([src_files[cg]], src_vocabs[p_(cg)[0]], None)
        for cg in cgs}

    # Create individial target streams
    trg_datasets = {
        cg: TextFile([trg_files[cg]], trg_vocabs[p_(cg)[1]], None)
        for cg in cgs}

    # Build the preprocessing pipeline for individual streams
    ind_streams = {}
    for cg in cgs:
        logger.info('Building training stream for cg:[{}]'.format(cg))
        masked_stream = get_src_trg_stream(
            cg, config, src_datasets, trg_datasets)
        ind_streams[cg] = masked_stream

    # Scheduler and meta-controller
    multi_enc_stream = MultiEncStream(
        ind_streams, schedule=config['schedule'],
        batch_sizes=config['batch_sizes'], transpose=True,
        start_after=config.get('start_after', None))
    return multi_enc_stream


# Regular pipeline for single source-target nmt stream
def get_src_trg_stream(cg, config, src_datasets=None, trg_datasets=None,
                       is_training=True, src_vocabs=None, trg_vocabs=None,
                       logprob_datasets=None):
    eid, did = p_(cg)
    if is_training:
        logger.info(' ... src:[{}] - [{}]'.format(
            eid, src_datasets[cg].files[0]))
        logger.info(' ... trg:[{}] - [{}]'.format(
            did, trg_datasets[cg].files[0]))
        stream = Merge([src_datasets[cg].get_example_stream(),
                        trg_datasets[cg].get_example_stream()],
                       ('source', 'target'))
        stream = Filter(stream, predicate=_too_long(config['seq_len']))

        if 'min_seq_lens' in config and config['min_seq_lens'][cg] > 0:
            stream = Filter(stream,
                            predicate=_too_short(config['min_seq_lens'][cg]))

        stream = Mapping(stream, _oov_to_unk(
                         src_vocab_size=config['src_vocab_sizes'][eid],
                         trg_vocab_size=config['trg_vocab_sizes'][did],
                         unk_id=config['unk_id']))
        stream = Batch(
            stream, iteration_scheme=ConstantScheme(
                config['batch_sizes'][cg]*config['sort_k_batches']))

        stream = Mapping(stream, SortMapping(_length))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(
            config['batch_sizes'][cg]))
    else:  # logprob stream
        src_dataset = TextFile([logprob_datasets[cg][0]],
                               src_vocabs[p_(cg)[0]], None)
        trg_dataset = TextFile([logprob_datasets[cg][1]],
                               trg_vocabs[p_(cg)[1]], None)
        stream = Merge([src_dataset.get_example_stream(),
                        trg_dataset.get_example_stream()],
                       ('source', 'target'))
        stream = Mapping(stream, _oov_to_unk(
                         src_vocab_size=config['src_vocab_sizes'][eid],
                         trg_vocab_size=config['trg_vocab_sizes'][did],
                         unk_id=config['unk_id']))
        bs = 100
        if 'log_prob_bs' in config:
            if isinstance(config['log_prob_bs'], dict):
                bs = config['log_prob_bs'][cg]
            else:
                bs = config['log_prob_bs']
        stream = Batch(stream, iteration_scheme=ConstantScheme(bs))

    masked_stream = Padding(stream)
    masked_stream = Mapping(
        masked_stream, _remapWordIdx(
            [(0, 0, config['src_eos_idxs'][eid]),
             (2, 0, config['trg_eos_idxs'][did])]))
    return masked_stream


def get_dev_streams(config):
    """Setup development set stream if necessary."""
    dev_streams = {}
    for cg in config['cgs']:
        if 'val_sets' in config and cg in config['val_sets']:
            logger.info('Building development stream for cg:[{}]'.format(cg))
            eid = p_(cg)[0]
            dev_file = config['val_sets'][cg]

            # Get dictionary and fix EOS
            dictionary = cPickle.load(open(config['src_vocabs'][eid]))
            dictionary['<S>'] = 0
            dictionary['<UNK>'] = config['unk_id']
            dictionary['</S>'] = config['src_eos_idxs'][eid]

            # Get as a text file and convert it into a stream
            dev_dataset = TextFile([dev_file], dictionary, None)
            dev_streams[cg] = DataStream(dev_dataset)
    return dev_streams


def get_logprob_streams(config):
    if 'log_prob_sets' not in config:
        return None

    cgs = config['cgs']
    enc_ids, dec_ids = get_enc_dec_ids(cgs)
    datasets = config['log_prob_sets']

    # Prepare source vocabs and files, make sure special tokens are there
    src_vocabs = {k: cPickle.load(open(v))
                  for k, v in config['src_vocabs'].iteritems()}
    for k in src_vocabs.keys():
        src_vocabs[k]['<S>'] = 0
        src_vocabs[k]['</S>'] = config['src_eos_idxs'][k]
        src_vocabs[k]['<UNK>'] = config['unk_id']

    # Prepare target vocabs and files, make sure special tokens are there
    trg_vocabs = {k: cPickle.load(open(v))
                  for k, v in config['trg_vocabs'].iteritems()}
    for k in trg_vocabs.keys():
        trg_vocabs[k]['<S>'] = 0
        trg_vocabs[k]['</S>'] = config['trg_eos_idxs'][k]
        trg_vocabs[k]['<UNK>'] = config['unk_id']

    # Build the preprocessing pipeline for individual streams
    ind_streams = {}
    for cg in cgs:
        eid, did = p_(cg)
        if cg not in datasets:
            continue
        logger.info('Building logprob stream for cg:[{}]'.format(cg))
        src_dataset = TextFile([datasets[cg][0]], src_vocabs[p_(cg)[0]], None)
        trg_dataset = TextFile([datasets[cg][1]], trg_vocabs[p_(cg)[1]], None)
        stream = Merge([src_dataset.get_example_stream(),
                        trg_dataset.get_example_stream()],
                       ('source', 'target'))

        stream = Mapping(stream, _oov_to_unk(
                         src_vocab_size=config['src_vocab_sizes'][eid],
                         trg_vocab_size=config['trg_vocab_sizes'][did],
                         unk_id=config['unk_id']))
        bs = 100
        if 'log_prob_bs' in config:
            if isinstance(config['log_prob_bs'], dict):
                bs = config['log_prob_bs'][cg]
            else:
                bs = config['log_prob_bs']

        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(
                bs, num_examples=get_num_lines(datasets[cg][0])))

        masked_stream = Padding(stream)
        masked_stream = Mapping(
            masked_stream, _remapWordIdx(
                [(0, 0, config['src_eos_idxs'][eid]),
                 (2, 0, config['trg_eos_idxs'][did])]))
        ind_streams[cg] = masked_stream

    return ind_streams


def get_log_prob_stream(cg, config):
    eid, did = p_(cg)
    dataset = config['log_prob_sets'][cg]

    # Prepare source vocabs and files, make sure special tokens are there
    src_vocab = cPickle.load(open(config['src_vocabs'][eid]))
    src_vocab['<S>'] = 0
    src_vocab['</S>'] = config['src_eos_idxs'][eid]
    src_vocab['<UNK>'] = config['unk_id']

    # Prepare target vocabs and files, make sure special tokens are there
    trg_vocab = cPickle.load(open(config['trg_vocabs'][did]))
    trg_vocab['<S>'] = 0
    trg_vocab['</S>'] = config['trg_eos_idxs'][did]
    trg_vocab['<UNK>'] = config['unk_id']

    # Build the preprocessing pipeline for individual streams
    logger.info('Building logprob stream for cg:[{}]'.format(cg))
    src_dataset = TextFile([dataset[0]], src_vocab, None)
    trg_dataset = TextFile([dataset[1]], trg_vocab, None)
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    stream = Mapping(stream, _oov_to_unk(
                     src_vocab_size=config['src_vocab_sizes'][eid],
                     trg_vocab_size=config['trg_vocab_sizes'][did],
                     unk_id=config['unk_id']))
    bs = 100
    if 'log_prob_bs' in config:
        if isinstance(config['log_prob_bs'], dict):
            bs = config['log_prob_bs'][cg]
        else:
            bs = config['log_prob_bs']
    stream = Batch(
        stream,
        iteration_scheme=ConstantScheme(
            bs, num_examples=get_num_lines(dataset[0])))

    masked_stream = Padding(stream)
    masked_stream = Mapping(
        masked_stream, _remapWordIdx(
            [(0, 0, config['src_eos_idxs'][eid]),
             (2, 0, config['trg_eos_idxs'][did])]))

    return masked_stream
