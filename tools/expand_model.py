import sys
import numpy

import logging

rng = numpy.random.RandomState(1234)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('expand_model')

ref_model = dict(numpy.load(sys.argv[1]))
old_model = dict(numpy.load(sys.argv[2]))

for o_key, o_val in old_model.items():
    if o_key not in ref_model:
        logger.info(
            'parameter {} does not exist in the old model'.format(o_key))
        continue
    elif o_val.shape != ref_model[o_key].shape:
        if len(set(o_val.shape) & set(ref_model[o_key].shape)) == 0 and \
                len(o_val.shape) > 1:
            logger.info(
                'not initializing {} since old shape{} and new shape {}'
                .format(o_key, o_val.shape, ref_model[o_key].shape))
        else:
            logger.info(
                'expanding parameter {} from {} into {}'.format(
                    o_key, o_val.shape, ref_model[o_key].shape))

            val = o_val.std() * rng.randn(*ref_model[o_key].shape).astype('float32') + \
                o_val.mean()
            if len(val.shape) == 1:
                val[:o_val.shape[0]] = o_val
            else:
                val[:o_val.shape[0], :o_val.shape[1]] = o_val

            ref_model[o_key] = val
    else:
        logger.info('copying parameter {}'.format(o_key))
        ref_model[o_key] = o_val

logger.info('saving {}.expanded'.format(sys.argv[1]))
numpy.savez(sys.argv[1] + '.expanded', **ref_model)
