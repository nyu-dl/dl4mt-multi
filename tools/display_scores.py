import argparse
import os
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=str, help="Directory of models")
    parser.add_argument("--ref-cg", type=str,
                        help="Reference pair to ")
    return parser.parse_args()


def get_scores(bleu_filenames, cg_names):

    def parse_bleu(fname):
        return float(re.search('(?<=BLEU)[.0-9]+', fname).group())

    def parse_iter(fname):
        return int(re.search('(?<=iter)[0-9]+', fname).group())

    scores = {xx: list() for xx in cg_names}
    for fname in bleu_filenames:
        cg_name = [nn for nn in cg_names if fname.find(nn) > 0][0]
        scores[cg_name].append((parse_bleu(fname), parse_iter(fname)))
    return scores


def print_scores(scores, cg_names, ref_cg):
    if ref_cg is None:
        for cg_name in cg_names:
            bleu, iter_ = max(scores[cg_name], key=lambda item: item[0])
            print 'Best score for [{}]: {} at iter {}'.format(cg_name, bleu,
                                                              iter_)
    else:
        bleu, iter_ = max(scores[ref_cg], key=lambda item: item[0])
        print 'Best score for [{}]: {} at iter {}'.format(ref_cg, bleu, iter_)
        for cg_name in cg_names:
            if cg_name != ref_cg:
                bleu = [bb for bb, tt in scores[cg_name] if tt == iter_][0]
                print 'Best score for [{}]: {} at iter {}'.format(
                    cg_name, bleu, iter_)


def get_bleu_filenames(model_dir):
    return [xx for xx in os.listdir(model_dir)
            if re.search('(?<=BLEU)[.0-9]+', xx) is not None
            and '_test_out' not in xx]


def get_cg_names(files_list):
    return list(set(
        [el.group(1) for el in
         filter(None, [re.search('validation_out_(.*).txt', xx)
                       for xx in files_list])]))

if __name__ == '__main__':

    args = parse_args()

    bleu_files = get_bleu_filenames(args.models)
    cg_names = get_cg_names(bleu_files)
    scores = get_scores(bleu_files, cg_names)
    print_scores(scores, cg_names, args.ref_cg)
