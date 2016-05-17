import sys
import numpy

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('replace_params')

ref_model = dict(numpy.load(sys.argv[1]))
old_model = dict(numpy.load(sys.argv[2]))

n_params_ref = len(ref_model.keys())
n_params_old = len(old_model.keys())
n_params_copied = 0

for o_key, o_val in old_model.items():
    if o_key not in ref_model:
        logger.info(
            'parameter {} does not exist in the old model, adding'.format(o_key))
    elif o_val.shape != ref_model[o_key].shape:
        raise ValueError(
            'shape mismatch: from{} to{}'.format(o_val.shape, ref_model[o_key].shape))
    else:
        logger.info('copying parameter {}'.format(o_key))
    ref_model[o_key] = o_val
    n_params_copied += 1

logger.info('saving {}.overwritten'.format(sys.argv[1]))
numpy.savez(sys.argv[1] + '.overwritten', **ref_model)
logger.info('number of params from   [{}]'.format(n_params_old))
logger.info('number of params to     [{}]'.format(n_params_ref))
logger.info('number of params copied [{}]'.format(n_params_copied))
