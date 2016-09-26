"""
Multi-way NMT setup and training loop.
"""
import logging
import os
import theano

from collections import OrderedDict
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from blocks.extensions import Printing, FinishAfter, Timing
from blocks.model import Model

from mcg.models import EncoderDecoder, MultiEncoder, MultiDecoder
from mcg.utils import p_, get_enc_dec_ids, get_version
from mcg.sampling import Sampler
from mcg.algorithm import SGDMultiCG, MainLoopWithMultiCGnoBlocks
from mcg.extensions import (
        CostMonitoringWithMultiCG, DumpWithMultiCG, LoadFromDumpMultiCG,
        PrintMultiStream, LogProbComputer, IncrementalDump)

logger = logging.getLogger(__name__)


def train(config, tr_stream, dev_stream, logprob_stream):

    trng = RandomStreams(config['seed'] if 'seed' in config else 1234)
    enc_ids, dec_ids = get_enc_dec_ids(config['cgs'])

    # Create Theano variables
    floatX = theano.config.floatX
    src_sel = tensor.matrix('src_selector', dtype=floatX)
    trg_sel = tensor.matrix('trg_selector', dtype=floatX)
    x = tensor.lmatrix('source')
    y = tensor.lmatrix('target')
    x_mask = tensor.matrix('source_mask')
    y_mask = tensor.matrix('target_mask')
    x_sampling = tensor.matrix('source', dtype='int64')
    y_sampling = tensor.vector('target', dtype='int64')
    prev_state = tensor.matrix('prev_state', dtype=floatX)
    src_sel_sampling = tensor.matrix('src_selector', dtype=floatX)
    trg_sel_sampling = tensor.matrix('trg_selector', dtype=floatX)

    # Create encoder-decoder architecture
    enc_dec = EncoderDecoder(
        encoder=MultiEncoder(enc_ids=enc_ids, **config),
        decoder=MultiDecoder(**config))

    # Build training computational graphs
    probs, opt_rets = enc_dec.build_models(
        x, x_mask, y, y_mask, src_sel, trg_sel)

    # Get costs
    costs = enc_dec.get_costs(probs, y, y_mask,
                              decay_cs=config.get('decay_c', None),
                              opt_rets=opt_rets)

    # Computation graphs
    cgs = enc_dec.get_computational_graphs(costs)

    # Build sampling models
    f_inits, f_nexts, f_next_states = enc_dec.build_sampling_models(
        x_sampling, y_sampling, src_sel_sampling, trg_sel_sampling, prev_state,
        trng=trng)

    # Some printing
    enc_dec.print_params(cgs)

    # Get training parameters with optional excludes
    training_params, excluded_params = enc_dec.get_training_params(
        cgs, exclude_encs=config['exclude_encs'],
        exclude_embs=config['exclude_embs'],
        additional_excludes=config['additional_excludes'],
        readout_only=config.get('readout_only', None),
        train_shared=config.get('train_shared', None))

    # Some more printing
    enc_dec.print_training_params(cgs, training_params)

    # Set up training algorithm
    algorithm = SGDMultiCG(
        costs=costs, tparams=training_params, drop_input=config['drop_input'],
        step_rule=config['step_rule'], learning_rate=config['learning_rate'],
        clip_c=config['step_clipping'],
        step_rule_kwargs=config.get('step_rule_kwargs', {}))

    # Set up training model
    training_models = OrderedDict()
    for k, v in costs.iteritems():
        training_models[k] = Model(costs[k])

    # Set extensions
    extensions = [
        Timing(after_batch=True),
        FinishAfter(after_n_batches=config['finish_after']),
        CostMonitoringWithMultiCG(after_batch=True),
        Printing(after_batch=True),
        PrintMultiStream(after_batch=True),
        DumpWithMultiCG(saveto=config['saveto'],
                        save_accumulators=config['save_accumulators'],
                        every_n_batches=config['save_freq'],
                        no_blocks=True)]

    # Reload model if necessary
    if config['reload'] and os.path.exists(config['saveto']):
        extensions.append(
            LoadFromDumpMultiCG(saveto=config['saveto'],
                                load_accumulators=config['load_accumulators'],
                                no_blocks=True))

    # Add sampling to computational graphs
    for i, (cg_name, cg) in enumerate(cgs.iteritems()):
        eid, did = p_(cg_name)
        if config['hook_samples'] > 0:
            extensions.append(Sampler(
                f_init=f_inits[cg_name], f_next=f_nexts[cg_name],
                data_stream=tr_stream, num_samples=config['hook_samples'],
                src_eos_idx=config['src_eos_idxs'][eid],
                trg_eos_idx=config['trg_eos_idxs'][did],
                enc_id=eid, dec_id=did,
                every_n_batches=config['sampling_freq'],
                cond_init_trg=config.get('cond_init_trg', False),
                f_next_state=f_next_states.get(cg_name, None)))

    # Save parameters incrementally without overwriting
    if config.get('incremental_dump', False):
        extensions.append(
            IncrementalDump(saveto=config['saveto'],
                            burnin=config['val_burn_in'],
                            every_n_batches=config['save_freq']))

    # Compute log probability on dev set
    if 'log_prob_freq' in config:
        extensions.append(
            LogProbComputer(
                cgs=config['cgs'],
                f_log_probs=enc_dec.build_f_log_probs(
                    probs, x, x_mask, y, y_mask, src_sel, trg_sel),
                streams=logprob_stream,
                every_n_batches=config['log_prob_freq']))

    # Initialize main loop
    main_loop = MainLoopWithMultiCGnoBlocks(
        models=training_models,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions,
        num_encs=config['num_encs'],
        num_decs=config['num_decs'])

    # Train!
    main_loop.run()

    # Be patient, after a month :-)
    print 'done'
