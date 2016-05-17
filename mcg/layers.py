import numpy
import theano

from .utils import _p, norm_weight, ortho_weight, concatenate
from theano import tensor

layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ff_init': ('param_init_ffinit_layer', 'ffinit_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond_multiEnc_v08': ('param_init_gru_cond_multiEnc_v08',
                                    'gru_cond_multiEnc_layer_v08')
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def relu(x):
    return tensor.maximum(0.0, x)


def logistic(x):
    return tensor.nnet.sigmoid(x)


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(params, prefix='ff', nin=None, nout=None, ortho=True,
                       add_bias=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if add_bias:
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def fflayer(tparams, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', add_bias=True, **kwargs):
    preact = tensor.dot(state_below, tparams[_p(prefix, 'W')])
    if add_bias:
        preact += tparams[_p(prefix, 'b')]
    return eval(activ)(preact)


# feedforward decoder initializer layer:
# two affine transformations (first one is applied by the encoder already,
#   the second one is shared across decoders) +
# point-wise nonlinearity +
# two affine transformations back (first one is shared across decoders,
#   the second one is decoder specific)
def param_init_ffinit_layer(params, prefix='ff_init', nin=None, nmid=None,
                            ncode=None, nout=None, ortho=True):

    if nmid is None:
        nmid = nin
    if ncode is None:
        ncode = nin

    # from encoder specific context embedding to code
    params[_p(prefix, 'W_shared')] = norm_weight(nin, ncode, scale=0.01,
                                                 ortho=ortho)
    # from code to decoder specific embedding
    params[_p(prefix, 'U_shared')] = norm_weight(ncode, nmid, scale=0.01,
                                                 ortho=ortho)

    # decoder specific embedding
    params[_p(prefix, 'U')] = norm_weight(nmid, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'c')] = numpy.zeros((nout,)).astype('float32')
    return params


def ffinit_layer(tparams, state_below, prefix='ff_init',
                 activ='lambda x: tensor.tanh(x)',
                 post_activ='lambda x: x', **kwargs):

    preact = tensor.dot(state_below, tparams[_p(prefix, 'W_shared')])
    code = eval(activ)(preact)
    act = tensor.dot(code, tparams[_p(prefix, 'U_shared')])

    act = tensor.dot(act, tparams[_p(prefix, 'U')])
    act += tparams[_p(prefix, 'c')]
    return eval(post_activ)(act)


# GRU layer
def param_init_gru(params, prefix='gru', nin=None, dim=None, hiero=False):
    if not hiero:
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W
        params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(
        state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, _ = theano.scan(_step,
                          sequences=seqs,
                          outputs_info=[tensor.alloc(0., n_samples, dim)],
                          non_sequences=[tparams[_p(prefix, 'U')],
                                         tparams[_p(prefix, 'Ux')]],
                          name=_p(prefix, '_layers'),
                          n_steps=nsteps,
                          strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention for multiple encoders - version 0.8
def param_init_gru_cond_multiEnc_v08(params, nin, dim, dimctx, dimatt,
                                     prefix='gru_cond_multiEnc_v08',
                                     **kwargs):
    """Note that, all the parameters with _att suffix are shared across
    decoders."""

    params = param_init_gru(params, prefix, nin=nin, dim=dim)

    # context to LSTM [C]
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    # context to LSTM [Cz, Cr]
    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden [Wi_dec]
    Wi_dec = norm_weight(nin, dimctx)
    params[_p(prefix, 'Wi_dec')] = Wi_dec

    # attention: LSTM -> hidden [Wa]
    Wd_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'Wd_dec')] = Wd_att

    # attention: hidden bias [ba]
    b_att = numpy.zeros((dimatt,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention: [va]
    U_att = norm_weight(dimatt, 1)
    params[_p(prefix, 'U_att')] = U_att

    # attention: encoder latent space embedder, [Le_att]
    Le_att = norm_weight(dimctx, dimatt).astype('float32')
    params[_p(prefix, 'Le_att')] = Le_att

    # attention: decoder latent space embedder, [Ld_att]
    Ld_att = norm_weight(dimctx, dimatt).astype('float32')
    params[_p(prefix, 'Ld_att')] = Ld_att

    # attention: weighted averages to gru input, post weight [Wp_att]
    Wp_att = norm_weight(dimctx, dimctx, ortho=False).astype('float32')
    params[_p(prefix, 'Wp_att')] = Wp_att

    return params


def gru_cond_multiEnc_layer_v08(
        tparams, state_below, state_before=None, mask=None, context=None,
        one_step=False, init_state=None, context_mask=None,
        prefix='gru_cond_multiEnc_v08', lencoder_act='tanh',
        ldecoder_act='tanh', **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3d: #annotation x #sample x dim + num_encs + num_decs'

    # for look, generate and update, the shifted emb is state_before not
    # state_below, so we set it accordingly here. If state_before is provided
    # we infer we are using look, generate and update
    if state_before is None:
        state_before = state_below

    # if we are using multiple latent spaces, previous word embedder for
    # decoder attention is not shared among decoders, it is decoder specific,
    # otherwise it is shared among all decoders. Parameter sharing is handled
    # by checking the postfix _att in the parameter list.
    state_belowc = tensor.dot(state_before, tparams[_p(prefix, 'Wi_dec')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # embed encoders latent space into attention outside of the scan
    enc_lspace = eval(lencoder_act)(
        context.dot(tparams[_p(prefix, 'Le_att')]) +
        tparams[_p(prefix, 'b_att')])

    # m_     : mask
    # x_     : state_below_ from target embeddings
    # xx_    : state_belowx from target embeddings
    # xc_    : state_belowc from source context
    # h_     : previous hidden state
    # ctx_   : previous weighted averages
    # alpha_ : previous weights
    # enc_ls : enc_lspace + b_att - encoder latent space
    # cc_    : context
    #              |    sequences   |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_,
                    enc_ls, cc_,
                    U, Wc, Wd_dec, U_att, Ux, Wcx, Ld_att, Wp_att):
        # attention
        # previous gru hidden state s_{i-1}
        pstate_ = tensor.dot(h_, Wd_dec)
        dec_lspace = eval(ldecoder_act)(xc_ + pstate_)

        # transform decoder latent space with shared attention parameters
        dec_lspace = dec_lspace.dot(Ld_att)

        # combine encoder and decoder latent spaces and compute alignments
        pctx__ = tensor.tanh(dec_lspace[None, :, :] + enc_ls)
        alpha = tensor.dot(pctx__, U_att)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # stabilize energies first
        max_alpha = alpha.max(axis=0)
        alpha = alpha - max_alpha
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        # project current context using a shared post projection
        ctx_ = ctx_.dot(Wp_att)

        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_dec')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'Ld_att')],
                   tparams[_p(prefix, 'Wp_att')]]

    pselectors = []

    if one_step:
        rval = _step(*(seqs +
                       [init_state] +
                       [enc_lspace, context] + shared_vars + pselectors))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state, None, None],
            non_sequences=[enc_lspace, context] + shared_vars + pselectors,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True)
    return rval
