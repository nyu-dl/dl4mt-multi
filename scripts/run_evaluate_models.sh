#!/bin/bash

modelsdir=${HOME}/trainedModels

THEANO_FLAGS=device=cpu,floatX=float32 \
    python ${HOME}/codes/dl4mt-multi/evaluate_models.py \
        --model-dir=${modelsdir}/multiWay \
        --num-process=6 \
        --proto=get_config_multiWay \
        --normalize

