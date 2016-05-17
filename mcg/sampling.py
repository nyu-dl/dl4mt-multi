import logging
import copy
import numpy
import operator
import os
import re
import signal
import time
import theano

from blocks.extensions import SimpleExtension

from collections import OrderedDict
from subprocess import Popen, PIPE
from toolz import merge

from .utils import _p, get_enc_dec_ids

logger = logging.getLogger(__name__)


def gen_sample(f_init, f_next, x, src_selector, trg_selector, k=1,
               maxlen=30, stochastic=True, argmax=False, eos_idx=0,
               cond_init_trg=False, ignore_unk=False, minlen=1, unk_idx=1,
               f_next_state=None, return_alphas=False):
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_decalphas = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_decalphas = []
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # multi-source
    inp_xs = [x]
    init_inps = inp_xs

    ret = f_init(*init_inps)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in range(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])

        prev_w = copy.copy(next_w)
        prev_state = copy.copy(next_state)
        inps = [next_w, ctx, next_state]

        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if return_alphas:
            next_decalpha = ret.pop(0)

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == eos_idx:
                break
        else:
            log_probs = numpy.log(next_p)

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, unk_idx] = -numpy.inf
            if ii < minlen:
                log_probs[:, eos_idx] = -numpy.inf

            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_decalphas = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

                if return_alphas:
                    tmp_decalphas = []
                    if ii > 0:
                        tmp_decalphas = copy.copy(hyp_decalphas[ti])
                    tmp_decalphas.append(next_decalpha[ti])
                    new_hyp_decalphas.append(tmp_decalphas)

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_decalphas = []

            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == eos_idx:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    if return_alphas:
                        sample_decalphas.append(new_hyp_decalphas[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if return_alphas:
                        hyp_decalphas.append(new_hyp_decalphas[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                if return_alphas:
                    sample_decalphas.append(hyp_decalphas[idx])

    if not return_alphas:
        return numpy.array(sample), numpy.array(sample_score)
    return numpy.array(sample), numpy.array(sample_score), \
        numpy.array(sample_decalphas)


class SamplingBase(object):

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, eos_idx):
        try:
            return seq.tolist().index(eos_idx) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq):
        return [x if x < self.src_vocab_size else self.unk_idx
                for x in seq]

    def _parse_input(self, line, eos_idx):
        seqin = line.split()
        seqlen = len(seqin)
        seq = numpy.zeros(seqlen+1, dtype='int64')
        for idx, sx in enumerate(seqin):
            seq[idx] = self.vocab.get(sx, self.unk_idx)
            if seq[idx] >= self.src_vocab_size:
                seq[idx] = self.unk_idx
        seq[-1] = eos_idx
        return seq

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _get_true_seq(self, seq, eos_idx):
        return seq[:self._get_true_length(seq, eos_idx)]

    def _make_matrix(self, arr):
        if arr.ndim >= 2:
            return arr
        return arr[None, :]


class Sampler(SimpleExtension, SamplingBase):
    """Samples from computation graph

        Does not use peeked batches
    """

    def __init__(self, f_init, f_next, data_stream, num_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, enc_id=0, dec_id=0, src_eos_idx=-1,
                 trg_eos_idx=-1, cond_init_trg=False, f_next_state=None,
                 **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.f_init = f_init
        self.f_next = f_next
        self.f_next_state = f_next_state
        self.data_stream = data_stream
        self.num_samples = num_samples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.cond_init_trg = cond_init_trg
        self.enc_id = enc_id
        self.dec_id = dec_id
        self._synced = False
        self.sampling_fn = gen_sample

    def do(self, which_callback, *args):

        batch = args[0]

        # Get current model parameters
        if not self._synced:
            sources = self._get_attr_rec(
                self.main_loop.data_stream.streams[_p(self.enc_id,
                                                      self.dec_id)],
                'data_stream')
            self.sources = sources
            self._synced = True

        batch = self.main_loop.data_stream\
            .get_batch_with_stream_id(_p(self.enc_id, self.dec_id))

        batch_size = batch['source'].shape[1]

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = self.sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = self.sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
            self.src_ivocab[self.src_eos_idx] = '</S>'
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.trg_ivocab[self.trg_eos_idx] = '</S>'

        sample_idx = numpy.random.choice(
            batch_size, self.num_samples, replace=False)
        src_batch = batch['source']
        trg_batch = batch['target']

        input_ = src_batch[:, sample_idx]
        target_ = trg_batch[:, sample_idx]

        # Sample
        outputs = [list() for _ in sample_idx]
        costs = [list() for _ in sample_idx]

        for i, idx in enumerate(sample_idx):
            outputs[i], costs[i] = self.sampling_fn(
                self.f_init, self.f_next, eos_idx=self.trg_eos_idx,
                x=self._get_true_seq(input_[:, i], self.src_eos_idx)[:, None],
                src_selector=self._make_matrix(batch['src_selector'][idx, :]),
                trg_selector=self._make_matrix(batch['trg_selector'][idx, :]),
                k=1, maxlen=30, stochastic=True, argmax=False,
                cond_init_trg=self.cond_init_trg,
                f_next_state=self.f_next_state)

        print ""
        logger.info("Sampling from computation graph[{}-{}]"
                    .format(self.enc_id, self.dec_id))
        for i in range(len(outputs)):
            input_length = self._get_true_length(input_[:, i],
                                                 self.src_eos_idx)
            target_length = self._get_true_length(target_[:, i],
                                                  self.trg_eos_idx)
            sample_length = self._get_true_length(outputs[i],
                                                  self.trg_eos_idx)

            print "Input : ", self._idx_to_word(input_[:, i][:input_length],
                                                self.src_ivocab)
            print "Target: ", self._idx_to_word(target_[:, i][:target_length],
                                                self.trg_ivocab)
            print "Sample: ", self._idx_to_word(outputs[i][:sample_length],
                                                self.trg_ivocab)
            print "Sample cost: ", costs[i].sum()
            print ""


class BleuValidator(SimpleExtension, SamplingBase):
    """Highly not recommended for use."""

    def __init__(self, f_init, f_next, data_stream,
                 bleu_script, val_set_out, val_set_grndtruth, src_vocab_size,
                 src_selector=None, trg_selector=None, n_best=1,
                 track_n_models=1, trg_ivocab=None, beam_size=5,
                 val_burn_in=10000, _reload=True, enc_id=None, dec_id=None,
                 saveto=None, src_eos_idx=-1, trg_eos_idx=-1, normalize=True,
                 cond_init_trg=False,**kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.f_init = f_init
        self.f_next = f_next
        self.data_stream = data_stream
        self.bleu_script = bleu_script
        self.val_set_out = val_set_out
        self.val_set_grndtruth = val_set_grndtruth
        self.src_vocab_size = src_vocab_size
        self.src_selector = src_selector
        self.trg_selector = trg_selector
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.trg_ivocab = trg_ivocab
        self.beam_size = beam_size
        self.val_burn_in = val_burn_in
        self._reload = _reload
        self.enc_id = enc_id
        self.dec_id = dec_id
        self.saveto = saveto if saveto else "."
        self.verbose = val_set_out
        self._synced = False

        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.normalize = normalize
        self.cond_init_trg = cond_init_trg

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.sampling_fn = gen_sample
        self.multibleu_cmd = ['perl', bleu_script, val_set_grndtruth, '<']

        # Create saving directory if it does not exist
        if not os.path.exists(saveto):
            os.makedirs(saveto)

        if self._reload:
            try:
                bleu_score = numpy.load(
                    os.path.join(
                        saveto, 'val_bleu_scores{}_{}.npz'.format(
                            self.enc_id, self.dec_id)))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.val_burn_in:
            return

        # Get current model parameters
        if not self._synced:
            enc_ids, dec_ids = get_enc_dec_ids(self.main_loop.models.keys())
            self.enc_idx = enc_ids.index(self.enc_id)
            self.dec_idx = dec_ids.index(self.dec_id)
            self.sources = self._get_attr_rec(
                self.main_loop.data_stream.streams[_p(self.enc_id,
                                                      self.dec_id)],
                'data_stream')
            self._synced = True

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        # Get target vocabulary
        if not self.trg_ivocab:
            trg_vocab = self.sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        if self.verbose:
            ftrans = open(self.val_set_out, 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = numpy.array(self._oov_to_unk(line[0])).astype('int64')

            # Branch for multiple computation graphs
            src_selector_input = numpy.zeros(
                (1, self.main_loop.num_encs)).astype(theano.config.floatX)
            src_selector_input[0, self.enc_idx] = 1.
            trg_selector_input = numpy.zeros(
                (1, self.main_loop.num_decs)).astype(theano.config.floatX)
            trg_selector_input[0, self.dec_idx] = 1.

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = self.sampling_fn(
                self.f_init, self.f_next,
                x=seq.reshape([len(seq), 1]), eos_idx=self.trg_eos_idx,
                src_selector=src_selector_input,
                trg_selector=trg_selector_input,
                k=self.beam_size, maxlen=3*len(seq), stochastic=False,
                argmax=False, cond_init_trg=self.cond_init_trg)

            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out[:-1],
                                                  self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print >> mb_subprocess.stdin, trans_out
                    if self.verbose:
                        print >> ftrans, trans_out

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info('BLEU score: {}'.format(bleu_score))
        mb_subprocess.terminate()

        # Save bleu scores to file
        self._save_bleu_scores()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(
                bleu_score, self.saveto, self.enc_id, self.dec_id)

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            params_to_save = []
            for cg_name in self.main_loop.models.keys():
                params_to_save.append(
                    self.main_loop.models[cg_name].get_param_values())
            params_to_save = merge(params_to_save)

            self._save_params(model, params_to_save)
            self._save_bleu_scores()

            signal.signal(signal.SIGINT, s)

    def _save_params(self, model, params):

        # Rename accordingly for blocks compatibility
        params_to_save = dict(
            (k.replace('/', '-'), v) for k, v in params.items())

        numpy.savez(model.path, **params_to_save)

    def _save_bleu_scores(self):
        numpy.savez(
            os.path.join(
                self.saveto,
                'val_bleu_scores{}_{}.npz'.format(self.enc_id, self.dec_id)),
            bleu_scores=self.val_bleu_curve)


class ModelInfo:
    def __init__(self, bleu_score, path=None, enc_id=None, dec_id=None):
        self.bleu_score = bleu_score
        self.enc_id = enc_id if enc_id is not None else ''
        self.dec_id = dec_id if dec_id is not None else ''
        self.path = self._generate_path(path) if path else None

    def _generate_path(self, path):
        return os.path.join(
            path, 'best_bleu_model{}_{}_{}_BLEU{:.2f}.npz'.format(
                self.enc_id, self.dec_id, int(time.time()), self.bleu_score))
