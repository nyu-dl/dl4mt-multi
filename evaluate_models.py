#!usr/bin/python
"""
Evaluate all the models in a directory.
Checks the new models in the directory and launches the decoding
script (translate.py) for each, one at a time. After no new models
found in the directory for 24 hours, exits.
"""
import os
import sys
import numpy as np
import time
import re
import subprocess
import signal
import argparse
import traceback

LIMIT_ITER = np.inf
SHUT_DOWN = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", 
			help="configuration file for translation script")
    parser.add_argument("-m", "--model-dir",
                        help="Parameter list for bleu script")
    parser.add_argument('--num-process', '-p', type=int, default=10)
    parser.add_argument('--proto', type=str)
    parser.add_argument('--normalize', '-n', action="store_false",
                        default=True)
    parser.add_argument("-i", "--nIter", type=int, default=LIMIT_ITER,
                        help="Number of iterations")
    parser.add_argument("--eval-script", '-e', type=str,
                        default="~/codes/dl4mt-multi/translate.py")
    return parser.parse_args()


def sigint_handler(_signo, _stack_frame):
    '''
    handles keyboard interrupt, ctrl+c
    '''
    print 'process interrupted'
    global SHUT_DOWN
    SHUT_DOWN = True


def call_script(model_path, config, proto, eval_script,
                num_process=10, normalize=True):
    '''
    open pipe and call the script
    '''
    try:
        subprocess.call(
            " python {} {} --config={} "
            " --proto={} "
            " -p {} {} {}".format(
                eval_script, model_path, config,
                proto, num_process,
                ' -n ' if normalize else ''),
            shell=True)
    except:
        traceback.print_exc(file=sys.stdout)
        print 'error in call_bleu_script()'
    print model_path


def sort_by_iter(files):
    """Ensures the evaluation start from the last saved model."""
    return sorted(
        files, key=lambda x: (int(re.findall(r'iter([^|]+).npz',x)[0])),
        reverse=True)


def get_iter_filenames(model_dir):
    return [xx for xx in os.listdir(model_dir)
            if re.search('(?<=iter)[.0-9]+.npz', xx) is not None]

if __name__ == "__main__":

    args = parse_args()

    signal.signal(signal.SIGINT, sigint_handler)

    lastModified = 0
    counter = 0
    sleep_counter = 0
    controllerSleep = 10 * 60  # check every 10 minutes
    items_to_eval = get_iter_filenames(args.model_dir)
    items_evaluated = []
    while not SHUT_DOWN and counter < args.nIter:
        items_to_eval = sort_by_iter(items_to_eval)
        for item in items_to_eval:
            print 'LAUNCHING JOB FOR MODEL: {}'.format(item)
            call_script(os.path.join(args.model_dir, item), args.proto,
                        args.eval_script, args.num_process, args.normalize)
        items_evaluated += items_to_eval
        items_to_eval = list(
            set(get_iter_filenames(args.model_dir)) - set(items_evaluated))
        if len(items_to_eval) == 0:
            print 'NO NEW ITEMS TO EVALUATE, SLEEPING...'
            time.sleep(controllerSleep)
            sleep_counter += 1
        else:
            sleep_counter = 0

        # Finish if no new items for a day
        if sleep_counter > 144:
            break
        counter += 1
    print 'done'
