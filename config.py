from collections import OrderedDict

from mcg.utils import (get_enc_dec_ids, get_paths, get_odict,
                       get_val_set_outs, ReadOnlyDict)

src_vocabs = {
    'en': 'all2en/all.tok.apos.clean.all-en.en.noDup.wmt15.joint.bpe30k.vocab.pkl',
    'de': 'de2en/all.tok.apos.clean.shuf.de-en.de.bpe30k.vocab.pkl',
    'fi': 'fi2en/all.tok.apos.clean.shuf.fi-en.fi.bpe30k.vocab.pkl'}

trg_vocabs = {
    'en': 'all2en/all.tok.apos.clean.all-en.en.noDup.wmt15.joint.bpe30k.vocab.pkl',
    'de': 'de2en/all.tok.apos.clean.shuf.de-en.de.bpe30k.vocab.pkl',
    'fi': 'fi2en/all.tok.apos.clean.shuf.fi-en.fi.bpe30k.vocab.pkl'
}

src_datas = {
    'en_de': 'de2en/all.tok.apos.clean.shuf.de-en.en.wmt15.joint.bpe30k',
    'en_fi': 'fi2en/all.tok.apos.clean.shuf.fi-en.en.wmt15.joint.bpe30k',
    'de_en': 'de2en/all.tok.apos.clean.shuf.de-en.de.bpe30k',
    'fi_en': 'fi2en/all.tok.apos.clean.shuf.fi-en.fi.bpe30k'
}

trg_datas = {
    'en_de': 'de2en/all.tok.apos.clean.shuf.de-en.de.bpe30k',
    'en_fi': 'fi2en/all.tok.apos.clean.shuf.fi-en.fi.bpe30k',
    'de_en': 'de2en/all.tok.apos.clean.shuf.de-en.en.wmt15.joint.bpe30k',
    'fi_en': 'fi2en/all.tok.apos.clean.shuf.fi-en.en.wmt15.joint.bpe30k'
}

val_sets_src = {
    'en_de': 'dev/all2en/newstest2013.en.tok.apos.wmt15.joint.bpe30k',
    'en_fi': 'dev/all2en/newsdev2015.en.tok.apos.wmt15.joint.bpe30k',
    'de_en': 'dev/de2en/newstest2013.de.tok.apos.bpe30k',
    'fi_en': 'dev/fi2en/newsdev2015.fi.tok.apos.bpe30k'
}

val_sets_ref = {
    'en_de': 'dev/de2en/newstest2013.de.tok.apos',
    'en_fi': 'dev/fi2en/newsdev2015.fi.tok.apos',
    'de_en': 'dev/all2en/newstest2013.en.tok.apos',
    'fi_en': 'dev/all2en/newsdev2015.en.tok.apos'
}

log_prob_sets = {
    'en_de': ['dev/all2en/newstest2013.en.tok.apos.wmt15.joint.bpe30k',
              'dev/de2en/newstest2013.de.tok.apos.bpe30k'],
    'en_fi': ['dev/all2en/newsdev2015.en.tok.apos.wmt15.joint.bpe30k',
              'dev/fi2en/newsdev2015.fi.tok.apos.bpe30k'],
    'de_en': ['dev/de2en/newstest2013.de.tok.apos.bpe30k',
              'dev/all2en/newstest2013.en.tok.apos.wmt15.joint.bpe30k'],
    'fi_en': ['dev/fi2en/newsdev2015.fi.tok.apos.bpe30k',
              'dev/all2en/newsdev2015.en.tok.apos.wmt15.joint.bpe30k']
}


def prototype_config_singleCG_08():
    config = {}
    config['num_encs'] = 1
    config['num_decs'] = 1
    config['src_seq_len'] = 50
    config['tgt_seq_len'] = 50
    config['representation_dim'] = 1200  # joint annotation dimension

    # Additional options for the model
    config['take_last'] = True
    config['multi_latent'] = True
    config['readout_dim'] = 1000
    config['representation_act'] = 'linear'  # encoder representation act
    config['lencoder_act'] = 'tanh'  # att-encoder latent space act
    config['ldecoder_act'] = 'tanh'  # att-decoder latent space act
    config['look_generate_update'] = False
    config['dec_rnn_type'] = 'gru_cond_multiEnc_v08'
    config['finit_mid_dim'] = 600
    config['finit_code_dim'] = 500
    config['finit_act'] = 'tanh'
    config['att_dim'] = 800

    # Optimization related
    config['sort_k_batches'] = 12
    config['step_rule'] = 'uAdam'
    config['learning_rate'] = 1e-4
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['save_accumulators'] = True
    config['load_accumulators'] = True

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    config['stream'] = 'multiCG_stream'
    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['track_n_models'] = 3
    config['output_val_set'] = True
    config['beam_size'] = 12

    # Validation set for log probs related
    config['log_prob_freq'] = 2000
    config['log_prob_bs'] = 10

    # Timing related
    config['reload'] = True
    config['save_freq'] = 5000
    config['sampling_freq'] = 17
    config['bleu_val_freq'] = 10000000
    config['val_burn_in'] = 60000
    config['finish_after'] = 10000000
    config['incremental_dump'] = True

    # Monitoring related
    config['hook_samples'] = 0
    config['plot'] = False
    config['bokeh_port'] = 3333
    return config


def prototype_config_multiCG_08(cgs):

    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Model related
    config = {}
    config['cgs'] = cgs
    config['num_encs'] = len(enc_ids)
    config['num_decs'] = len(dec_ids)
    config['src_seq_len'] = 50
    config['tgt_seq_len'] = 50
    config['representation_dim'] = 1200  # joint annotation dimension
    config['enc_nhids'] = get_odict(enc_ids, 1000)
    config['dec_nhids'] = get_odict(dec_ids, 1000)
    config['enc_embed_sizes'] = get_odict(enc_ids, 620)
    config['dec_embed_sizes'] = get_odict(dec_ids, 620)

    # Additional options for the model
    config['take_last'] = True
    config['multi_latent'] = True
    config['readout_dim'] = 1000
    config['representation_act'] = 'linear'  # encoder representation act
    config['lencoder_act'] = 'tanh'  # att-encoder latent space act
    config['ldecoder_act'] = 'tanh'  # att-decoder latent space act
    config['dec_rnn_type'] = 'gru_cond_multiEnc_v08'
    config['finit_mid_dim'] = 600
    config['finit_code_dim'] = 500
    config['finit_act'] = 'tanh'
    config['att_dim'] = 1200

    # Optimization related
    config['batch_sizes'] = get_odict(cgs, 60)
    config['sort_k_batches'] = 12
    config['step_rule'] = 'uAdam'
    config['learning_rate'] = 2e-4
    config['step_clipping'] = 1
    config['weight_scale'] = 0.01
    config['schedule'] = get_odict(cgs, 1)
    config['save_accumulators'] = True  # algorithms' update step variables
    config['load_accumulators'] = True  # be careful with this
    config['exclude_encs'] = get_odict(enc_ids, False)
    config['exclude_embs'] = False
    config['min_seq_lens'] = get_odict(cgs, 0)
    config['additional_excludes'] = get_odict(cgs, [])

    # Regularization related
    config['drop_input'] = get_odict(cgs, 0.)
    config['decay_c'] = get_odict(cgs, 0.)
    config['alpha_c'] = get_odict(cgs, 0.)
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary related
    config['src_vocab_sizes'] = get_odict(enc_ids, 30000)
    config['trg_vocab_sizes'] = get_odict(dec_ids, 30000)
    config['src_eos_idxs'] = get_odict(enc_ids, 0)
    config['trg_eos_idxs'] = get_odict(dec_ids, 0)
    config['stream'] = 'multiCG_stream'
    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['track_n_models'] = 3
    config['output_val_set'] = True
    config['beam_size'] = 12

    # Validation set for log probs related
    config['log_prob_freq'] = 2000
    config['log_prob_bs'] = 10

    # Timing related
    config['reload'] = True
    config['save_freq'] = 10000
    config['sampling_freq'] = 17
    config['bleu_val_freq'] = 10000000
    config['val_burn_in'] = 1
    config['finish_after'] = 2000000
    config['incremental_dump'] = True

    # Monitoring related
    config['hook_samples'] = 2
    config['plot'] = False
    config['bokeh_port'] = 3333

    return config


def get_config_single():

    cgs = ['de_en']
    config = prototype_config_multiCG_08(cgs)
    enc_ids, dec_ids = get_enc_dec_ids(cgs)
    config['saveto'] = 'single'

    basedir = ''
    config['batch_sizes'] = OrderedDict([('de_en', 80)])
    config['schedule'] = OrderedDict([('de_en', 12)])
    config['src_vocabs'] = get_paths(enc_ids, src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, trg_vocabs, basedir)
    config['src_datas'] = get_paths(cgs, src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, trg_datas, basedir)
    config['save_freq'] = 5000
    config['val_burn_in'] = 60000
    config['bleu_script'] = basedir + '/multi-bleu.perl'
    config['val_sets'] = get_paths(cgs, val_sets_src, basedir)
    config['val_set_grndtruths'] = get_paths(cgs, val_sets_ref, basedir)
    config['val_set_outs'] = get_val_set_outs(config['cgs'], config['saveto'])
    config['log_prob_sets'] = get_paths(cgs, log_prob_sets, basedir)

    return ReadOnlyDict(config)


def get_config_multiEncoder():

    cgs = ['fi_en', 'de_en']
    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Model related
    config = prototype_config_multiCG_08(cgs)
    config['saveto'] = 'multiEncoder'

    # Vocabulary/dataset related
    basedir = ''
    config['src_vocabs'] = get_paths(enc_ids, src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, trg_vocabs, basedir)
    config['src_datas'] = get_paths(cgs, src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['bleu_script'] = basedir + '/multi-bleu.perl'
    config['val_sets'] = get_paths(cgs, val_sets_src, basedir)
    config['val_set_grndtruths'] = get_paths(cgs, val_sets_ref, basedir)
    config['val_set_outs'] = get_val_set_outs(config['cgs'], config['saveto'])
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, log_prob_sets, basedir)

    return ReadOnlyDict(config)


def get_config_multiDecoder():

    cgs = ['en_fi', 'en_de']
    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Model related
    config = prototype_config_multiCG_08(cgs)
    config['saveto'] = 'multiDec'

    # Vocabulary/dataset related
    basedir = ''
    config['src_vocabs'] = get_paths(enc_ids, src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, trg_vocabs, basedir)
    config['src_datas'] = get_paths(cgs, src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['bleu_script'] = basedir + '/multi-bleu.perl'
    config['val_sets'] = get_paths(cgs, val_sets_src, basedir)
    config['val_set_grndtruths'] = get_paths(cgs, val_sets_ref, basedir)
    config['val_set_outs'] = get_val_set_outs(config['cgs'], config['saveto'])
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, log_prob_sets, basedir)

    return ReadOnlyDict(config)


def get_config_multiWay():

    cgs = ['fi_en', 'de_en', 'en_de']
    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Model related
    config = prototype_config_multiCG_08(cgs)
    config['saveto'] = 'multiWay'

    # Vocabulary/dataset related
    basedir = ''
    config['src_vocabs'] = get_paths(enc_ids, src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, trg_vocabs, basedir)
    config['src_datas'] = get_paths(cgs, src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['bleu_script'] = basedir + '/multi-bleu.perl'
    config['val_sets'] = get_paths(cgs, val_sets_src, basedir)
    config['val_set_grndtruths'] = get_paths(cgs, val_sets_ref, basedir)
    config['val_set_outs'] = get_val_set_outs(config['cgs'], config['saveto'])
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, log_prob_sets, basedir)

    return ReadOnlyDict(config)
