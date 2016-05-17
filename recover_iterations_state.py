import argparse
import logging
import os
import pprint

import config_dcc as cfg

from blocks.extensions import FinishAfter
from blocks.serialization import secure_pickle_dump

from mcg.stream import get_tr_stream
from mcg.algorithm import MainLoopWithMultiCGnoBlocks

logger = logging.getLogger(__name__)


class DummyAlgorithm(object):
    '''Simulate training by doing nothing.'''
    def __init__(self, **kwargs):
        self.dummy_iter = 0

    def process_batch(self, batch, **kwargs):
        self.dummy_iter += 1
        if self.dummy_iter % 1000 == 0:
            logger.info(self.dummy_iter)

    def initialize(self, **kwargs):
        pass


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",
                        default="get_config_refMultiCG_de_en_06_BPE",
                        help="Prototype config to use for model configuration")
    parser.add_argument('--iters', type=int, default=0,
                        help="Iterations done in total for all cgs, default 0 \
                        eg. --iters=1000")
    return parser


if __name__ == "__main__":
    # Get the arguments
    args = get_parser().parse_args()

    config = getattr(cfg, args.proto)()

    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    tr_stream = get_tr_stream(config)

    logger.info("Will iterate up to iteration: [{}]".format(args.iters))

    extensions = [FinishAfter(after_n_batches=args.iters)]

    # Initialize main loop
    main_loop = MainLoopWithMultiCGnoBlocks(
        models=[None for _ in config['cgs']],
        algorithm=DummyAlgorithm(),
        data_stream=tr_stream,
        extensions=extensions,
        num_encs=config['num_encs'],
        num_decs=config['num_decs'])

    # Run dummy main-loop
    logger.info(" ...running dummy main-loop")
    main_loop.run()

    logger.info(" ...saving iteration state")
    path_to_iteration_state = os.path.join(config['saveto'],
                                           'iterations_state.pkl')
    if os.path.exists(path_to_iteration_state):
        logger.warn('Iteration state already exists! appending .new')
        path_to_iteration_state += '.new'
    secure_pickle_dump(main_loop.iteration_state,
                       path_to_iteration_state)
