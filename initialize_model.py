"""
Initializes multi-way model using pre-trained parameters.
"""
import argparse
import logging
import numpy
import os
import theano

import config_mSrc as configuration

from theano import tensor

from mcg.models import (
    EncoderDecoder, MultiEncoder, MultiDecoder)
from mcg.utils import get_enc_dec_ids


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('initialize_params')


def get_parser():

    def dict_type(ss):
        return dict([map(str.strip, s.split(':'))
                     for s in ss.split(',')])

    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str)
    parser.add_argument('--ref-encs', type=dict_type,
                        help="Models to initialize encoders, \
                        eg. --ref-encs=fi:file1,de:file2")
    parser.add_argument('--ref-decs', type=dict_type,
                        help="Models to initialize decoders, \
                        eg. --ref-decs=en:file1,de:file2")
    parser.add_argument('--ref-att', type=str,
                        help="Model to initialize shared components")
    parser.add_argument('--ref-dec-embs', type=dict_type,
                        help="Models to initialize decoder embeddings, \
                        eg. --ref-dec-embs=en:file1,de:file2")
    parser.add_argument('--ref-enc-embs', type=dict_type,
                        help="Models to initialize encoder embeddings, \
                        eg. --ref-enc-embs=en:file1,de:file2")
    return parser


def tparams_asdict(tparams):
    d = {}
    for tname, tparam in tparams.items():
        d[tname] = tparam.get_value()
    return d


def set_tparam(tparam, param):
    tshape = tparam.get_value().shape
    pshape = param.shape
    if tshape != pshape:
        val = tparam.get_value().copy()
        if tshape[0] > pshape[0]:
            val[:pshape[0]] = param
        else:
            val = param.copy()
        logger.warn(
            "Dimension mismatch [{}]:{} - reference {}"
            .format(tparam.name, tshape, pshape))
    else:
        val = param.copy()
    tparam.set_value(val)


def main(config, ref_encs=None, ref_decs=None, ref_att=None,
         ref_enc_embs=None, ref_dec_embs=None):

    # Create Theano variables
    floatX = theano.config.floatX
    src_sel = tensor.matrix('src_selector', dtype=floatX)
    trg_sel = tensor.matrix('trg_selector', dtype=floatX)
    x = tensor.lmatrix('source')
    y = tensor.lmatrix('target')
    x_mask = tensor.matrix('source_mask')
    y_mask = tensor.matrix('target_mask')

    # for multi source - maximum is 5 for now
    xs = [tensor.lmatrix('source%d' % i) for i in range(5)]
    x_masks = [tensor.matrix('source%d_mask' % i) for i in range(5)]

    # Create encoder-decoder architecture, and initialize
    logger.info('Creating encoder-decoder')
    enc_ids, dec_ids = get_enc_dec_ids(config['cgs'])
    enc_dec = EncoderDecoder(
        encoder=MultiEncoder(enc_ids=enc_ids, **config),
        decoder=MultiDecoder(**config))
    enc_dec.build_models(x, x_mask, y, y_mask, src_sel, trg_sel,
                         xs=xs, x_masks=x_masks)

    # load reference encoder models
    r_encs = {}
    if ref_encs is not None:
        for eid, path in ref_encs.items():
            logger.info('... ref-enc[{}] loading [{}]'.format(eid, path))
            r_encs[eid] = dict(numpy.load(path))

    # load reference decoder models
    r_decs = {}
    if ref_decs is not None:
        for did, path in ref_decs.items():
            logger.info('... ref-dec[{}] loading [{}]'.format(did, path))
            r_decs[did] = dict(numpy.load(path))

    # load reference model for the shared components
    if ref_att is not None:
        logger.info('... ref-shared loading [{}]'.format(ref_att))
        r_att = dict(numpy.load(ref_att))

    num_params_set = 0
    params_set = {k: 0 for k in enc_dec.get_params().keys()}

    # set encoder parameters of target model
    for eid, rparams in r_encs.items():
        logger.info(' Setting encoder [{}] parameters ...'.format(eid))
        tparams = enc_dec.encoder.encoders[eid].tparams
        for pname, pval in tparams.items():
            set_tparam(tparams[pname], rparams[pname])
            params_set[pname] += 1
            num_params_set += 1
        set_tparam(enc_dec.encoder.tparams['ctx_embedder_%s_W' % eid],
                   rparams['ctx_embedder_%s_W' % eid])
        set_tparam(enc_dec.encoder.tparams['ctx_embedder_%s_b' % eid],
                   rparams['ctx_embedder_%s_b' % eid])
        params_set['ctx_embedder_%s_W' % eid] += 1
        params_set['ctx_embedder_%s_b' % eid] += 1
        num_params_set += 2

    # set decoder parameters of target model
    for did, rparams in r_decs.items():
        logger.info(' Setting decoder [{}] parameters ...'.format(did))
        tparams = enc_dec.decoder.decoders[did].tparams
        for pname, pval in tparams.items():
            set_tparam(tparams[pname], rparams[pname])
            params_set[pname] += 1
            num_params_set += 1

    # set shared component parameters of target model
    if ref_att is not None:
        logger.info(' Setting shared parameters ...')
        shared_enc, shared_params = enc_dec.decoder._get_shared_params()
        for pname in shared_params.keys():
            set_tparam(enc_dec.decoder.tparams[pname], r_att[pname])
            params_set[pname] += 1
            num_params_set += 1

    # set encoder embeddings
    if ref_enc_embs is not None:
        logger.info(' Setting encoder embeddings ...')
        for eid, path in ref_enc_embs.items():
            pname = 'Wemb_%s' % eid
            logger.info(' ... [{}]-[{}]'.format(did, pname))
            emb = numpy.load(path)[pname]
            set_tparam(enc_dec.encoder.tparams[pname], emb)
            params_set[pname] += 1
            num_params_set += 1

    # set decoder embeddings
    if ref_dec_embs is not None:
        logger.info(' Setting decoder embeddings ...')
        for did, path in ref_dec_embs.items():
            pname = 'Wemb_dec_%s' % did
            logger.info(' ... [{}]-[{}]'.format(did, pname))
            emb = numpy.load(path)[pname]
            set_tparam(enc_dec.decoder.tparams[pname], emb)
            params_set[pname] += 1
            num_params_set += 1

    logger.info(' Saving initialized params to [{}/.params.npz]'
                .format(config['saveto']))
    if not os.path.exists(config['saveto']):
        os.makedirs(config['saveto'])

    numpy.savez('{}/params.npz'.format(config['saveto']),
                **tparams_asdict(enc_dec.get_params()))
    logger.info(' Total number of params    : [{}]'
                .format(len(enc_dec.get_params())))
    logger.info(' Total number of params set: [{}]'.format(num_params_set))
    logger.info(' Duplicates [{}]'.format(
        [k for k, v in params_set.items() if v > 1]))
    logger.info(' Unset (random) [{}]'.format(
        [k for k, v in params_set.items() if v == 0]))
    logger.info(' Set {}'.format(
        [k for k, v in params_set.items() if v > 0]))


if __name__ == "__main__":

    args = get_parser().parse_args()
    config = getattr(configuration, args.proto)().copy()
    main(config, args.ref_encs, args.ref_decs, args.ref_att,
         args.ref_enc_embs, args.ref_dec_embs)
