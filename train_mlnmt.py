import argparse
import logging
import pprint

from mlnmt import train
from mcg.stream import (get_tr_stream, get_dev_streams,
                        get_logprob_streams)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",
                        default="get_config_multiWay",
                        help="Prototype config to use for model configuration")
    args = parser.parse_args()

    import config as cfg
    config = getattr(cfg, args.proto)()

    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    train(config, get_tr_stream(config), get_dev_streams(config),
          get_logprob_streams(config))
