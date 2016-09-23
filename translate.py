"""
Decoding pipeline.
"""
import argparse
import cPickle
import logging
import numpy
import os
import pprint
import re
import theano
import time
import importlib


from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from mcg.models import EncoderDecoder, MultiEncoder, MultiDecoder
from mcg.sampling import gen_sample
from mcg.utils import get_enc_dec_ids, p_, seqs2words, words2seqs

from multiprocessing import Process, Queue
from subprocess import Popen, PIPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('translate')


def get_parser():

    def dict_type(ss):
        return dict([map(str.strip, s.split(':'))
                     for s in ss.split(',')])

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-process', '-p', type=int, default=5,
                        help="Number of process to use for decoding")
    parser.add_argument('--config', type=str,
                        help="model config for translation")
    parser.add_argument('--proto', type=str,
                        help="Model prototype from config")
    parser.add_argument('--normalize', '-n', action="store_true",
                        default=True, help="Normalize with seq-len")
    parser.add_argument('--char-level', '-c', action="store_true",
                        default=False, help="Character level")
    parser.add_argument('--cgs-to-translate', type=lambda s: s.split(','),
                        help='comma separeted string of cg names\
                        eg. --cgs-to-translate=fi_en,de_en')
    parser.add_argument('--n-best', type=int, default=1)
    parser.add_argument('--zero-shot', action="store_true", default=False,
                        help="Experimental")
    parser.add_argument('--test', action="store_true", default=False,
                        help="Append _test while decoding")
    parser.add_argument('--gold-files', type=dict_type,
                        help="Groundtruth files (optional), \
                        eg. --gold-files=fi_en:file1,de_en:file2")
    parser.add_argument('--source-files', type=dict_type,
                        help="Source files (optional), \
                        eg. --source-files=fi_en:file1,de_en:file2")
    parser.add_argument("--changes", type=dict_type,
                        help="Changes to config")
    parser.add_argument('model', type=str)
    return parser


def calculate_bleu(bleu_script, trans, gold):
    multibleu_cmd = ['perl', bleu_script, gold, '<']
    mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)
    print >> mb_subprocess.stdin, '\n'.join(trans)
    mb_subprocess.stdin.flush()
    mb_subprocess.stdin.close()
    stdout = mb_subprocess.stdout.readline()
    logger.info(stdout)
    out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
    assert out_parse is not None
    bleu_score = float(out_parse.group()[6:])
    mb_subprocess.terminate()
    return bleu_score


def _translate(seq, f_init, f_next, trg_eos_idx, src_sel, trg_sel,
               k, cond_init_trg, normalize, n_best, **kwargs):
    sample, score = gen_sample(
        f_init, f_next, x=numpy.array(seq).reshape([len(seq), 1]),
        eos_idx=trg_eos_idx, src_selector=src_sel, trg_selector=trg_sel,
        k=k, maxlen=3*len(seq), stochastic=False, argmax=False,
        cond_init_trg=cond_init_trg, **kwargs)
    if normalize:
        lengths = numpy.array([len(s) for s in sample])
        score = score / lengths
    if n_best == 1:
        sidx = numpy.argmin(score)
    elif n_best > 1:
        sidx = numpy.argsort(score)[:n_best]
    else:
        raise ValueError('n_best cannot be negative!')
    return sample[sidx], score[sidx]


def translate_model(queue, rqueue, pid, f_init, f_next, src_sel, trg_sel,
                    trg_eos_idx, k, normalize, cond_init_trg, n_best,
                    **kwargs):

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx
        seq, scores = _translate(x, f_init, f_next, trg_eos_idx, src_sel,
                                 trg_sel, k, cond_init_trg, normalize, n_best,
                                 **kwargs)

        rqueue.put((idx, seq, scores))

    return


def main(config, model, normalize=False, n_process=5, chr_level=False,
         cgs_to_translate=None, n_best=1, zero_shot=False, test=False):

    trng = RandomStreams(config['seed'] if 'seed' in config else 1234)
    enc_ids, dec_ids = get_enc_dec_ids(config['cgs'])
    iternum = re.search('(?<=iter)[0-9]+', model)

    # Translate only the chosen cgs if they are valid
    if cgs_to_translate is None:
        cgs_to_translate = config['cgs']

    # Check if computational graphs are valid
    if not set(config['cgs']) >= set(cgs_to_translate) and not zero_shot:
        raise ValueError('{} not a subset of {}!'.format(
            cgs_to_translate, config['cgs']))

    # Check if zero shot computational graph is valid
    if zero_shot:
        if len(cgs_to_translate) > 1:
            raise ValueError('Only one cg can be translated for zero shot')
        if p_(cgs_to_translate[0])[0] not in enc_ids or \
                p_(cgs_to_translate[0])[1] not in dec_ids:
            raise ValueError('Zero shot is not valid for {}'
                             .format(cgs_to_translate[0]))
        config['cgs'] += cgs_to_translate

    # Create Theano variables
    floatX = theano.config.floatX
    src_sel = tensor.matrix('src_selector', dtype=floatX)
    trg_sel = tensor.matrix('trg_selector', dtype=floatX)
    x_sampling = tensor.matrix('source', dtype='int64')
    y_sampling = tensor.vector('target', dtype='int64')
    prev_state = tensor.matrix('prev_state', dtype=floatX)

    # Create encoder-decoder architecture
    logger.info('Creating encoder-decoder')
    enc_dec = EncoderDecoder(
        encoder=MultiEncoder(enc_ids=enc_ids, **config),
        decoder=MultiDecoder(**config))

    # Allocate parameters
    enc_dec.init_params()

    # Build sampling models
    logger.info('Building sampling models')
    f_inits, f_nexts, f_next_states = enc_dec.build_sampling_models(
        x_sampling, y_sampling, src_sel, trg_sel, prev_state, trng=trng)

    # Load parameters
    logger.info('Loading parameters')
    enc_dec.load_params(model)

    # Output translation file names to be returned
    translations = {}

    # Iterate over computational graphs
    for cg_name in f_inits.keys():

        enc_name = p_(cg_name)[0]
        dec_name = p_(cg_name)[1]
        enc_idx = enc_ids.index(enc_name)
        dec_idx = dec_ids.index(dec_name)
        f_init = f_inits[cg_name]
        f_next = f_nexts[cg_name]
        f_next_state = f_next_states.get(cg_name, None)

        # For monolingual paths do not perform any translations
        if enc_name == dec_name or cg_name not in cgs_to_translate:
            logger.info('Passing the validation of computational graph [{}]'
                        .format(cg_name))
            continue

        logger.info('Validating computational graph [{}]'.format(cg_name))

        # Change output filename
        if zero_shot:
            config['val_set_outs'][cg_name] += '_zeroShot'

        # Get input and output file names
        source_file = config['val_sets'][cg_name]
        saveto = config['val_set_outs'][cg_name]
        saveto = saveto + '{}_{}'.format(
            '' if iternum is None else '_iter' + iternum.group(),
            'nbest' if n_best > 1 else 'BLEU')

        # pass if output exists
        if len([ff for ff in os.listdir(config['saveto'])
                if ff.startswith(os.path.basename(saveto))]):
            logger.info('Output file {}* exists, skipping'.format(saveto))
            continue

        # Prepare source vocabs and files, make sure special tokens are there
        src_vocab = cPickle.load(open(config['src_vocabs'][enc_name]))
        src_vocab['<S>'] = 0
        src_vocab['</S>'] = config['src_eos_idxs'][enc_name]
        src_vocab['<UNK>'] = config['unk_id']

        # Invert dictionary
        src_ivocab = dict()
        for kk, vv in src_vocab.iteritems():
            src_ivocab[vv] = kk

        # Prepare target vocabs and files, make sure special tokens are there
        trg_vocab = cPickle.load(open(config['trg_vocabs'][dec_name]))
        trg_vocab['<S>'] = 0
        trg_vocab['</S>'] = config['trg_eos_idxs'][dec_name]
        trg_vocab['<UNK>'] = config['unk_id']

        # Invert dictionary
        trg_ivocab = dict()
        for kk, vv in trg_vocab.iteritems():
            trg_ivocab[vv] = kk

        def _send_jobs(fname):
            with open(fname, 'r') as f:
                for idx, line in enumerate(f):
                    x = words2seqs(
                        line, src_vocab,
                        vocab_size=config['src_vocab_sizes'][enc_name],
                        chr_level=chr_level)
                    queue.put((idx, x))
            return idx+1

        def _finish_processes():
            for midx in xrange(n_process):
                queue.put(None)

        def _retrieve_jobs(n_samples):
            trans = [None] * n_samples
            scores = [None] * n_samples
            for idx in xrange(n_samples):
                resp = rqueue.get()
                trans[resp[0]] = resp[1]
                scores[resp[0]] = resp[2]
                if numpy.mod(idx, 10) == 0:
                    print 'Sample ', (idx+1), '/', n_samples, ' Done'
            return trans, scores

        # Create source and target selector vectors
        src_selector_input = numpy.zeros(
            (1, enc_dec.num_encs)).astype(theano.config.floatX)
        src_selector_input[0, enc_idx] = 1.
        trg_selector_input = numpy.zeros(
            (1, enc_dec.num_decs)).astype(theano.config.floatX)
        trg_selector_input[0, dec_idx] = 1.

        # Actual translation here
        logger.info('Translating ' + source_file + '...')
        val_start_time = time.time()
        if n_process == 1:
            trans = []
            scores = []
            with open(source_file, 'r') as f:
                for idx, line in enumerate(f):
                    if idx % 100 == 0 and idx != 0:
                        logger.info('...translated [{}] lines'.format(idx))
                    seq = words2seqs(
                        line, src_vocab,
                        vocab_size=config['src_vocab_sizes'][enc_name],
                        chr_level=chr_level)
                    _t, _s = _translate(
                        seq, f_init, f_next, trg_vocab['</S>'],
                        src_selector_input, trg_selector_input,
                        config['beam_size'],
                        config.get('cond_init_trg', False),
                        normalize, n_best, f_next_state=f_next_state)
                    trans.append(_t)
                    scores.append(_s)

        else:
            # Create queues
            queue = Queue()
            rqueue = Queue()
            processes = [None] * n_process
            for midx in xrange(n_process):
                processes[midx] = Process(
                    target=translate_model,
                    args=(queue, rqueue, midx, f_init, f_next,
                          src_selector_input, trg_selector_input,
                          trg_vocab['</S>'], config['beam_size'], normalize,
                          config.get('cond_init_trg', False), n_best),
                    kwargs={'f_next_state': f_next_state})
                processes[midx].start()

            n_samples = _send_jobs(source_file)
            trans, scores = _retrieve_jobs(n_samples)
            _finish_processes()

        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))

        # Prepare translation outputs and calculate BLEU if necessary
        # Note that, translations are post processed for BPE here
        if n_best == 1:
            trans = seqs2words(trans, trg_vocab, trg_ivocab)
            trans = [tt.replace('@@ ', '') for tt in trans]
            bleu_score = calculate_bleu(
                bleu_script=config['bleu_script'], trans=trans,
                gold=config['val_set_grndtruths'][cg_name])
            saveto += '{}'.format(bleu_score)
        else:
            n_best_trans = []
            for idx, (n_best_tr, score_) in enumerate(zip(trans, scores)):
                sentences = seqs2words(n_best_tr, trg_vocab, trg_ivocab)
                sentences = [tt.replace('@@ ', '') for tt in sentences]
                for ids, trans_ in enumerate(sentences):
                    n_best_trans.append(
                        '|||'.join(
                            ['{}'.format(idx), trans_,
                             '{}'.format(score_[ids])]))
            trans = n_best_trans

        # Write to file
        with open(saveto, 'w') as f:
            print >>f, '\n'.join(trans)
        translations[cg_name] = saveto
    return translations, saveto


if __name__ == "__main__":

    args = get_parser().parse_args()

    configuration = importlib.import_module(
        args.config.split('.')[0] if '.py' in args.config else args.config)
    config = getattr(configuration, args.proto)().copy()
    if args.changes is not None:
        config.update(args.changes)

    # Set source and gold files in config if provided
    if args.source_files is not None:
        for cg_name, s_file in args.source_files.items():
            config['val_sets'][cg_name] = s_file
            if args.test:
                config['val_set_outs'][cg_name] = \
                    os.path.join(config['saveto'],
                                 os.path.basename(s_file) + '_test_out')
            else:
                config['val_set_outs'][cg_name] = s_file + '_validation_out'
    if args.gold_files is not None:
        for cg_name, g_file in args.gold_files.items():
            config['val_set_grndtruths'][cg_name] = g_file

    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    main(config, args.model, normalize=args.normalize,
         n_process=args.num_process, chr_level=args.char_level,
         cgs_to_translate=args.cgs_to_translate, n_best=args.n_best,
         zero_shot=args.zero_shot, test=args.test)
