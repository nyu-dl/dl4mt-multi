
import cPickle
import logging
import numpy
import os
import re
import shutil
import signal
import sys
import tempfile
import theano
import time

from collections import OrderedDict
from toolz import merge

from blocks.dump import MainLoopDumpManager
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.extensions.saveload import LoadFromDump, Dump
from blocks.serialization import pickle_dump

from .utils import p_, get_enc_dec_ids
from .sampling import SamplingBase

logger = logging.getLogger(__name__)


def secure_pickle_dump(object_, path):
    """Try pickling into a temporary file and then move."""
    try:
        dirname = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dirname) as temp:
            pickle_dump(object_, temp)
        shutil.move(temp.name, path)
    except Exception as e:
        # if "temp" in locals():
        #    os.remove(temp.name)
        logger.error(" Error {0}".format(str(e)))


def secure_numpy_save(params_dict, path):
    """Try saving into a temporary file and then move."""
    try:
        dirname = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dirname) as temp:
            numpy.savez(temp, **params_dict)
        shutil.move(temp.name, path)
    except Exception as e:
        # if "temp" in locals():
        #    os.remove(temp.name)
        logger.error(" Error {0}".format(str(e)))


class PrintMultiStream(SimpleExtension):
    """Prints number of batches seen for each data stream"""
    def __init__(self, **kwargs):
        super(PrintMultiStream, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        counters = self.main_loop.data_stream.training_counter
        epochs = self.main_loop.data_stream.epoch_counter
        sid = self.main_loop.data_stream.curr_id
        src_size = args[0]['source'].shape
        trg_size = args[0]['target'].shape
        msg = ['Source_{}:iter[{}]-epoch[{}]'
               .format(name, counters[name], epochs[name])
               for name in counters.keys()]
        print("Multi-stream status:")
        print "\t", "Using stream: source_{}".format(sid)
        print "\t", "Source shape: {}".format(src_size)
        print "\t", "Target shape: {}".format(trg_size)
        print "\t", " ".join(msg)


class IncrementalDump(SimpleExtension, SamplingBase):
    """
    Incrementally dumps model given frequency.
    --------------------------------------------------------
    Saves the model parameters (params.npz), data stream iterations state
    (iteration_state.pkl) and training log (log) by appending the iteration
    number (_iterX).
    """

    def __init__(self, saveto, burnin=100000, save_iter_state=True,
                 save_log=True, **kwargs):
        super(IncrementalDump, self).__init__(**kwargs)
        self.saveto = saveto
        self.burnin = burnin
        self.save_iter_state = save_iter_state
        self.save_log = save_log
        self.modelID = int(time.time())  # self._get_model_id(saveto)

    def _get_model_id(self, saveto):
        try:
            postfix = [int(m.group(1))
                       for m in [re.match(r'.*_([-0-9]+)', f)
                                 for f in os.listdir(saveto)]
                       if m is not None]
            model_id = max(postfix)
        except:
            model_id = 0
        return model_id

    def do(self, which_callback, *args):
        iterations_done = self.main_loop.status['iterations_done']
        if self.burnin <= iterations_done:
            # Save the model here
            iterations_done = self.main_loop.status['iterations_done']
            filename = os.path.join(
                self.saveto, 'params_iter{}.npz'.format(iterations_done))
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info(" Incremental dump {}".format(filename))
            params_to_save = []
            for cg_name in self.main_loop.models.keys():
                params_to_save.append(
                    self.main_loop.models[cg_name].get_param_values())
            params_to_save = merge(params_to_save)
            secure_numpy_save(params_to_save, filename)
            if self.save_iter_state:
                filename_is = os.path.join(
                    self.saveto,
                    'iterations_state_iter{}.pkl'.format(iterations_done))
                logger.info(" Incremental dump {}".format(filename_is))
                secure_pickle_dump(self.main_loop.iteration_state, filename_is)
            if self.save_log:
                filename_log = os.path.join(
                    self.saveto,
                    'log_iter{}'.format(iterations_done))
                logger.info(" Incremental dump {}".format(filename_log))
                secure_pickle_dump(self.main_loop.log, filename_log)
            signal.signal(signal.SIGINT, s)


class MainLoopDumpManagerWMT15(MainLoopDumpManager):
    """
    Checkpointintg for multi CG main loop.
    --------------------------------------------------------
    Dump manager handles all the necessary operations for loading and
    saving a model, iteration state and training log. Algorithm step rule
    accumulators are also saved and loaded if specified.
    """

    def __init__(self, saveto, save_accumulators=False,
                 load_accumulators=False):
        super(MainLoopDumpManagerWMT15, self).__init__(saveto)
        self.save_accumulators = save_accumulators
        self.load_accumulators = load_accumulators

    @property
    def path_to_accumulators(self):
        return os.path.join(self.folder, 'algo{}.npz')

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            for name, model in main_loop.models.iteritems():
                params_this = model.get_params()
                missing = set(params_this) - set(params_all)
                for pname in params_this.keys():
                    if pname in params_all:
                        val = params_all[pname]
                        if params_this[pname].get_value().shape != val.shape:
                            logger.warning(
                                " Dimension mismatch {}-{} for {}"
                                .format(params_this[pname].get_value().shape,
                                        val.shape, pname))

                        params_this[pname].set_value(val)
                        logger.info(" Loaded to CG[{}] {:15}: {}"
                                    .format(name, val.shape, pname))
                    else:
                        logger.warning(
                            " Parameter does not exist: {}".format(pname))

                logger.info(
                    " Number of params loaded for computation graph[{}]: {}"
                    .format(name, len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
            sys.exit(1)

        try:
            logger.info(" Loading iteration state...")
            # TODO: this is a workaround for backward compatibility
            schedule = OrderedDict(main_loop.data_stream.schedule)
            main_loop.iteration_state = self.load_iteration_state()
            main_loop.data_stream.fix_schedule_after_reload(schedule)
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
            logger.warn("... trying to recover log")
            iters_done = sum(main_loop.data_stream.training_counter.values())
            if iters_done > 0:
                logger.warn("... setting iterations done to {}"
                            .format(iters_done))
                main_loop.status['iterations_done'] = iters_done
            else:
                logger.warn("... unsuccessful")

        if self.load_accumulators:
            try:
                logger.info(" Loading algorithm accumulators...")
                self._load_accumulators(main_loop)
            except Exception as e:
                logger.error(" Error {0}".format(str(e)))
                sys.exit(1)

    def dump_parameters(self, main_loop):
        params_to_save = []
        for model in main_loop.models.values():
            params_to_save.append(model.get_param_values())
        secure_numpy_save(merge(params_to_save),
                          self.path_to_parameters)

    def dump_accumulators(self, main_loop):
        """Each step rule has different number of accumulators"""
        for cg_name, model in main_loop.models.iteritems():
            algo = main_loop.algorithm.algorithms[cg_name]
            accums = algo.step_rule_updates
            params = algo.steps.items()
            model_params = model.get_params()

            # Reshape this long list into (num_params, num_accums_per_param)
            num_params = len(params)
            num_accums = len(accums)
            assert num_accums % num_params == 0, \
                "Accumulators cannot be loaded for CG[{}]".format(cg_name)

            # This is num_accums_per_param
            col = num_accums / num_params
            accums_mat = [accums[col*l:col*(l+1)] for l in range(num_params)]
            accums_vals = [[y[0].get_value() for y in x] for x in accums_mat]

            # Get corresponding parameter names and create a dictionary
            names = [[k for k, v in model_params.iteritems()
                      if v == params[l][0]][0] for l in xrange(len(params))]
            params_dict = dict([(names[l].replace("/", "-"), accums_vals[l])
                                for l in xrange(len(names))])

            # Save here
            secure_numpy_save(params_dict,
                              self.path_to_accumulators.format(cg_name))

    def dump(self, main_loop):
        """Overwrites MainLoopDumpManager.dump()."""
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        print ""
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        secure_pickle_dump(main_loop.iteration_state,
                           self.path_to_iteration_state)
        logger.info(" ...saving log")
        secure_pickle_dump(main_loop.log, self.path_to_log)
        if self.save_accumulators:
            logger.info(" ...saving algorithm")
            self.dump_accumulators(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def _load_accumulators(self, main_loop):
        """Nasty method, use carefully"""
        for cg_name, model in main_loop.models.iteritems():
            source = numpy.load(self.path_to_accumulators.format(cg_name))
            accums_dict = {name.replace("-", "/"): value
                           for name, value in source.items()}
            source.close()
            algo = main_loop.algorithm.algorithms[cg_name]
            model_params = model.get_params()
            steps = algo.steps.items()

            for pidx in xrange(len(steps)):
                # Get parameter name and its accumulators
                p = steps[pidx][0]
                name = [k for k, v in model_params.iteritems() if v == p][0]
                accums = accums_dict[name]

                # This is num_accums_per_param
                col = len(accums)
                for aidx in xrange(col):
                    algo.step_rule_updates[pidx*col+aidx][0].set_value(
                        accums[aidx])

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return cPickle.load(source)


class DumpWithMultiCG(Dump):
    """Wrapper to use MainLoopDumpManagerWMT15"""
    def __init__(self, saveto, save_accumulators=False, no_blocks=False,
                 **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpWithMultiCG, self).__init__(saveto, **kwargs)
        if no_blocks:
            self.manager = MainLoopDumpManagerNoBlocks(
                saveto=saveto, save_accumulators=save_accumulators)
        else:
            self.manager = MainLoopDumpManagerWMT15(
                saveto, save_accumulators=save_accumulators)


class LoadFromDumpMultiCG(LoadFromDump):
    """Wrapper to use MainLoopDumpManagerWMT15"""

    def __init__(self, saveto, load_accumulators=False, no_blocks=False,
                 **kwargs):
        super(LoadFromDumpMultiCG, self).__init__(saveto, **kwargs)
        if no_blocks:
            self.manager = MainLoopDumpManagerNoBlocks(
                saveto=saveto, load_accumulators=load_accumulators)
        else:
            self.manager = MainLoopDumpManagerWMT15(
                saveto, load_accumulators=load_accumulators)


class LogProbComputer(SimpleExtension, MonitoringExtension):
    """
    Computes log probabilities on a given set.
    --------------------------------------------------------
    For each computational graph, this extension computes the validation
    set log-probabilities. The data stream differs from training data stream
    as we do not use sorted batches. Note that, nll can be used as a proxy
    for early stopping, which is faster than approximate decoding for blue.
    """

    def __init__(self, cgs, f_log_probs, streams, **kwargs):
        super(LogProbComputer, self).__init__(**kwargs)
        self.cgs = cgs
        self.f_log_probs = f_log_probs
        self.streams = streams
        self.enc_ids, self.dec_ids = get_enc_dec_ids(self.cgs)
        self.num_encs = len(self.enc_ids)
        self.num_decs = len(self.dec_ids)

    def do(self, callback_name, *args):
        probs = {}
        print ''
        logger.info(" Computing log-probs...")
        start = time.time()
        for cg_name, stream in self.streams.iteritems():
            probs[cg_name] = list()
            src_id, trg_id = p_(cg_name)

            # handle multi-source stream
            src_idx = self.enc_ids.index(src_id)
            trg_idx = self.dec_ids.index(trg_id)

            for i, batch in enumerate(stream.get_epoch_iterator()):
                batch_size = batch[0].shape[0]
                src_sel = numpy.zeros(
                    (batch_size, self.num_encs)).astype(theano.config.floatX)
                src_sel[:, src_idx] = 1.
                trg_sel = numpy.zeros(
                    (batch_size, self.num_decs)).astype(theano.config.floatX)
                trg_sel[:, trg_idx] = 1.

                inps = [batch[0].T, batch[1].T, batch[2].T, batch[3].T,
                        src_sel, trg_sel]

                pprobs = self.f_log_probs[cg_name](*inps)
                probs[cg_name].append(pprobs.tolist())

                if numpy.isnan(numpy.mean(probs[cg_name])):
                    import ipdb
                    ipdb.set_trace()

            print 'logprob for CG [{}]: {}'.format(
                cg_name, numpy.mean(probs[cg_name]))

        print "took {} seconds.".format(time.time()-start)
        records = [('logprob_' + k, numpy.mean(v))
                   for k, v in probs.iteritems()]
        self.add_records(self.main_loop.log, records)


class CostMonitoringWithMultiCG(SimpleExtension, MonitoringExtension):
    """Fetches cost from algorithm and adds it to log."""

    def __init__(self, **kwargs):
        super(CostMonitoringWithMultiCG, self).__init__(**kwargs)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        if (self.main_loop.status['iterations_done'] ==
                self._last_time_called):
            raise Exception("TrainingDataMonitoring.do should be invoked"
                            " no more than once per iteration")
        self._last_time_called = self.main_loop.status['iterations_done']
        self.add_records(self.main_loop.log, [self.main_loop.algorithm._cost])


class MainLoopDumpManagerNoBlocks(MainLoopDumpManagerWMT15):
    """Checkpointintg for multi CG main loop."""

    def __init__(self, **kwargs):
        super(MainLoopDumpManagerNoBlocks, self).__init__(**kwargs)

    def dump_accumulators(self, main_loop):
        """Each step rule has different number of accumulators"""
        for cg_name, model in main_loop.models.iteritems():
            algo = main_loop.algorithm.algorithms[cg_name]
            accums = algo.step_rule_updates

            # Get corresponding parameter names and create a dictionary
            params_dict = dict([(acc.name, acc.get_value())
                                for acc in accums])
            # Save here
            secure_numpy_save(params_dict,
                              self.path_to_accumulators.format(cg_name))

    def _load_accumulators(self, main_loop):
        """Load accumulators with some checks."""
        for cg_name, model in main_loop.models.iteritems():

            # Load accumulators
            accum_filename = self.path_to_accumulators.format(cg_name)
            if not os.path.isfile(accum_filename):
                logger.error(" Accumulators file does not exist [{}]"
                             .format(accum_filename))
                continue

            source = numpy.load(accum_filename)
            accums_to_load = {k: v for k, v in source.items()}
            source.close()

            algo = main_loop.algorithm.algorithms[cg_name]
            accums = algo.step_rule_updates

            # Set accumulators
            for acc in accums:
                try:
                    acc.set_value(accums_to_load[acc.name])
                except:
                    logger.error(" Could not load {}".format(acc.name))
