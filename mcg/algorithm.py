
import logging
import numpy
import re
import signal
import theano
import time
import traceback

from collections import OrderedDict
from theano import tensor

from blocks import config as cfg
from blocks.extensions import CallbackName
from blocks.graph import ComputationGraph
from blocks.log import TrainingLog
from blocks.main_loop import (TrainingFinish,
                              error_in_error_handling_message,
                              error_message)
from blocks.utils import (change_recursion_limit, reraise_as, unpack)
from blocks.utils.profile import Profile, Timer

from .utils import get_enc_dec_ids, _p, itemlist, make_ordered_dict


logger = logging.getLogger(__name__)

epoch_interrupt_message = """

Blocks will complete this epoch iteration of training and run extensions \
before exiting. If you do not want to complete this epoch, press CTRL + C \
again to stop training after the current batch."""

batch_interrupt_message = """

Blocks will complete the current batch and run extensions before exiting. If \
you do not want to complete this batch, press CTRL + C again. WARNING: Note \
that this will end training immediately, and extensions that e.g. save your \
training progress won't be run."""


class SGDMultiCG(object):
    """
    Gradient Descent that trains only one CG at a time
    --------------------------------------------------------
    Computes the gradients of costs wrt model parameters, applies gradient
    clipping and non-finite fix. Step function (optimizer) is compiled here.
    """

    def __init__(self, costs, tparams, step_rule, drop_input=None,
                 learning_rate=None, clip_c=0., step_rule_kwargs=None,
                 **kwargs):
        """
        costs : dict, mapping cg_name to cost
        tparams : dict, mapping cg_name to shared parameters
        step_rule : str, optimizer
        drop_input : dict, mapping cg_name to drop_input ratio (float)
        learning_rate : theano tensor variable
        clip_c : float, gradient clipping threshold
        step_rule_kwargs : dict, additional arguments to the step rule
        """
        self.costs = costs
        self.tparams = tparams
        self.step_rule = step_rule
        self.learning_rate = learning_rate
        self.clip_c = clip_c
        self.step_rule_kwargs = step_rule_kwargs

        self.num_cgs = len(costs)
        self.cg_names = costs.keys()
        self.enc_ids, self.dec_ids = get_enc_dec_ids(self.cg_names)

        self.f_grads = OrderedDict()
        self.f_grad_shareds = OrderedDict()
        self.f_updates = OrderedDict()
        self.drop_input = drop_input
        self._cost = None
        self.algorithms = OrderedDict()  # blocks legacy

        if drop_input is None:
            self.drop_input = {name: 0.0 for name in costs.keys()}

        for cg_name in self.cg_names:
            cost = self.costs[cg_name]
            inps = ComputationGraph(cost).inputs
            params = make_ordered_dict(self.tparams[cg_name])
            logger.info(
                "Initializing the training algorithm [{}]".format(cg_name))

            logger.info("...computing gradient")
            grads = theano.tensor.grad(
                cost=cost, wrt=self.tparams[cg_name])

            if self.clip_c > 0.:
                logger.info("...clipping gradients")
                g2 = 0.
                for g in grads:
                    g2 += (g**2).sum()
                notfinite = tensor.isnan(g2) + tensor.isinf(g2)
                new_grads = []
                for g in grads:
                    p = self._get_p_from_g(cg_name, g, params)
                    tmpg = tensor.switch(
                        g2 > (self.clip_c**2),
                        g / tensor.sqrt(g2) * self.clip_c, g)
                    new_grads.append(
                        tensor.switch(notfinite, numpy.float32(.1) * p, tmpg))
                grads = new_grads

            start_time = time.time()
            logger.info("...building optimizer",)
            lr = tensor.scalar(name='lr')
            self.f_grad_shareds[cg_name], self.f_updates[cg_name], \
                step_rule_updates = eval(
                    self.step_rule)(lr, params, grads, inps, cost,
                                    **self.step_rule_kwargs)
            logger.info(" took: {} seconds".format(time.time() - start_time))

            # blocks legacy, just a helper
            self.algorithms[cg_name] = Algorithm(cost, inps, params, grads,
                                                 step_rule_updates)

    def initialize(self):
        pass

    def get_cg_id_from_selectors(self, src_selector, trg_selector):
        """
        During training, cg_name is inferred from source and target
        selector arrays, so this section is important.
        src_selector : numpy.array, encoder one hot indicator
        trg_selector : numpy.array, decoder one hot indicator
        """
        enc = '.'.join([self.enc_ids[i]
                        for i, sel in enumerate(src_selector) if sel])
        dec = '.'.join([self.dec_ids[i]
                        for i, sel in enumerate(trg_selector) if sel])
        return _p(enc, dec)

    def process_batch(self, batch):
        """
        Execution of an update step, infer cg_id from selectors, and pick
        corresponding computational graph, and apply batch to the CG.
        """
        cg_id = self.get_cg_id_from_selectors(batch['src_selector'][0],
                                              batch['trg_selector'][0])

        # Apply input replacement with <UNK> if necessary
        if self.drop_input[cg_id] > 0.0:
            num_els = numpy.prod(batch['source'].shape)
            num_reps = max(1, int(num_els * self.drop_input[cg_id]))
            replace_idx = numpy.random.choice(num_els, num_reps, replace=False)
            # TODO: set it according to unk_id in config
            batch['source'][numpy.unravel_index(
                replace_idx, batch['source'].shape)] = 1

        ordered_batch = [batch[v.name] for v in self.algorithms[cg_id].inputs]

        # To save memory, we may combine f_update and f_grad_shared
        if self.f_grad_shareds[cg_id] is None:
            inps = [self.learning_rate] + ordered_batch
            cost = self.f_updates[cg_id](*inps)
            self._cost = ('cost_' + cg_id, cost)
        else:
            cost = self.f_grad_shareds[cg_id](*ordered_batch)
            self._cost = ('cost_' + cg_id, cost)
            self.f_updates[cg_id](self.learning_rate)

    def _get_p_from_g(self, cg_id, g, params):
        """
        Utility function to pick the parameter given gradient.
        """
        p_name = re.search('\(dcost_' + cg_id + '/d(.+?)\)', g.name).group(1)
        return params[p_name]


def uAdam(lr, tparams, grads, inp, cost,
          b1=0.9, b2=0.999, e=1e-8, ups=None,
          no_gshared=False):

    if ups is not None:
        ups = list(ups.iteritems())
    else:
        ups = []

    # branch to use less memory, avoid allocating shared variables
    if no_gshared:
        logger.info("......saving memory, not allocating gshared!")
        gshared = grads
        f_grad_shared = None
    else:
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.iteritems()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]
        f_grad_shared = theano.function(inp, cost, updates=gsup+ups)

    updates = []
    step_rule_updates = []

    i = theano.shared(numpy.float32(0.), 'time')
    i_t = i + 1.
    fix1 = b1**(i_t)
    fix2 = b2**(i_t)
    lr_t = lr * (tensor.sqrt(1 - fix2) / (1 - fix1))

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')

        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * g**2
        m_t_hat = m_t / (1 - fix1)
        v_t_hat = v_t / (1 - fix2)
        g_t = lr_t * m_t_hat / (tensor.sqrt(v_t_hat) + e)
        p_t = p - g_t

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        step_rule_updates.append(m)
        step_rule_updates.append(v)

    updates.append((i, i_t))
    step_rule_updates.append(i)

    if no_gshared:
        f_update = theano.function([lr] + inp, cost, updates=updates+ups,
                                   on_unused_input='ignore')
    else:
        f_update = theano.function([lr], [], updates=updates,
                                   on_unused_input='ignore')
    return f_grad_shared, f_update, step_rule_updates


class MainLoopWithMultiCGnoBlocks(object):
    """
    Standalone MainLoop in order to handle multi CG without blocks.
    --------------------------------------------------------
    Main-loop represents the training loop and handles necessary actions
    before, during and after the training. Given a model, algorithm and
    data_stream, main loop feeds the algorithm with the batches from
    data_stream and updates the model. Main loop calls the defined extensions
    when they are needed. Also keeps a log which contains all the status.
    """

    def __init__(self, models, algorithm, data_stream,
                 num_encs=1, num_decs=1, log=None, extensions=None):
        """
        models : dict, mapping cg_name to blocks.model
        algorithm : SGDMultiCG
        data_stream : data stream, either MultiSourceStream or MultiEncStream
        num_encs : int, number of encoders
        num_decs : int, number of decoders
        log : blocks.log, the logger object
        extensions : blocks.extensions, the main loop extensions
        """
        self.models = models
        self.num_encs = num_encs
        self.num_decs = num_decs
        self.num_cgs = len(models)

        if log is None:
            log = TrainingLog()
        if extensions is None:
            extensions = []

        self.data_stream = data_stream
        self.algorithm = algorithm
        self.log = log
        self.extensions = extensions

        self.profile = Profile()

        self.status['training_started'] = False
        self.status['epoch_started'] = False
        self.status['epoch_interrupt_received'] = False
        self.status['batch_interrupt_received'] = False

    @property
    def iteration_state(self):
        """Quick access to the (data stream, epoch iterator) pair."""
        return (self.data_stream, self.epoch_iterator)

    @iteration_state.setter
    def iteration_state(self, value):
        (self.data_stream, self.epoch_iterator) = value

    @property
    def status(self):
        """A shortcut for `self.log.status`."""
        return self.log.status

    def run(self):
        logging.basicConfig()

        with change_recursion_limit(cfg.recursion_limit):
            self.original_sigint_handler = signal.signal(
                signal.SIGINT, self._handle_epoch_interrupt)
            self.original_sigterm_handler = signal.signal(
                signal.SIGTERM, self._handle_batch_interrupt)
            try:
                logger.info("Entered the main loop")
                if not self.status['training_started']:
                    for extension in self.extensions:
                        extension.main_loop = self
                    self._run_extensions('before_training')
                    with Timer('initialization', self.profile):
                        self.algorithm.initialize()
                    self.status['training_started'] = True
                if self.log.status['iterations_done'] > 0:
                    self._run_extensions('on_resumption')
                    self.status['epoch_interrupt_received'] = False
                    self.status['batch_interrupt_received'] = False
                with Timer('training', self.profile):
                    while self._run_epoch():
                        pass
            except TrainingFinish:
                self.log.current_row['training_finished'] = True
            except Exception as e:
                self._restore_signal_handlers()
                self.log.current_row['got_exception'] = traceback.format_exc(e)
                logger.error("Error occured during training." + error_message)
                try:
                    self._run_extensions('on_error')
                except Exception as inner_e:
                    logger.error(traceback.format_exc(inner_e))
                    logger.error("Error occured when running extensions." +
                                 error_in_error_handling_message)
                reraise_as(e)
            finally:
                if self.log.current_row.get('training_finished', False):
                    self._run_extensions('after_training')
                if cfg.profile:
                    self.profile.report()
                self._restore_signal_handlers()

    def find_extension(self, name):
        """Find an extension with a given name."""
        return unpack([extension for extension in self.extensions
                       if extension.name == name], singleton=True)

    def _run_epoch(self):
        if not self.status.get('epoch_started', False):
            try:
                self.log.status['received_first_batch'] = False
                self.epoch_iterator = (self.data_stream.
                                       get_epoch_iterator(as_dict=True))
            except StopIteration:
                return False
            self.status['epoch_started'] = True
            self._run_extensions('before_epoch')
        with Timer('epoch', self.profile):
            while self._run_iteration():
                pass
        self.status['epoch_started'] = False
        self.status['epochs_done'] += 1
        self.status['_epoch_ends'].append(self.status['iterations_done'])
        self._run_extensions('after_epoch')
        self._check_finish_training('epoch')
        return True

    def _run_iteration(self):
        try:
            with Timer('read_data', self.profile):
                batch = next(self.epoch_iterator)
        except StopIteration:
            if not self.log.status['received_first_batch']:
                reraise_as(ValueError("epoch iterator yielded zero batches"))
            return False
        self.log.status['received_first_batch'] = True
        self._run_extensions('before_batch', batch)
        with Timer('train', self.profile):
            self.algorithm.process_batch(batch)
        self.status['iterations_done'] += 1
        self._run_extensions('after_batch', batch)
        self._check_finish_training('batch')
        return True

    def _run_extensions(self, method_name, *args):
        with Timer(method_name, self.profile):
            for extension in self.extensions:
                with Timer(type(extension).__name__, self.profile):
                    extension.dispatch(CallbackName(method_name), *args)

    def _check_finish_training(self, level):
        """Checks whether the current training should be terminated."""
        # In case when keyboard interrupt is handled right at the end of
        # the iteration the corresponding log record can be found only in
        # the previous row.
        if (self.log.current_row.get('training_finish_requested', False) or
                self.status.get('batch_interrupt_received', False)):
            raise TrainingFinish
        if (level == 'epoch' and
                self.status.get('epoch_interrupt_received', False)):
            raise TrainingFinish

    def _handle_epoch_interrupt(self, signal_number, frame):
        # Try to complete the current epoch if user presses CTRL + C
        logger.warning('Received epoch interrupt signal.' +
                       epoch_interrupt_message)
        signal.signal(signal.SIGINT, self._handle_batch_interrupt)
        self.log.current_row['epoch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['epoch_interrupt_received'] = True

    def _handle_batch_interrupt(self, signal_number, frame):
        # After 2nd CTRL + C or SIGTERM signal (from cluster) finish batch
        self._restore_signal_handlers()
        logger.warning('Received batch interrupt signal.' +
                       batch_interrupt_message)
        self.log.current_row['batch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['batch_interrupt_received'] = True

    def _restore_signal_handlers(self):
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)


class Algorithm(object):
    """Legacy class, a placeholder for blocks.algorithm."""
    def __init__(self, cost, inputs, params, grads, step_rule_updates):
        self.cost = cost
        self.inputs = inputs
        self.steps = params
        self.grads = grads
        self.step_rule_updates = step_rule_updates
