# Multi-Way Neural Machine Translation
This repo implements multi-way Neural Machine Translation described in the paper 
"Multi-way Multilingual Neural Machine Translation with a Shared Attention Mechanism". 
In NAACL,2016.

With this repo, you can build a multi-encoder, multi-decoder or a multi-way NMT model. 
When you reduce the number of encoders and decoders to one respectively, you basically retain a single-pair NMT model with attention mechanism.<br>

Dependencies:
-------------
The code consists of three major components for dependencies:
 1. Core computational graphs ([Theano](https://github.com/Theano/Theano))
 2. Data streams ([Fuel](https://github.com/mila-udem/fuel))
 3. Training loop and extensions ([Blocks](https://github.com/mila-udem/blocks))

Please use `setup.sh` for setting up your development environment.<br>


Navigation:
-------------
The core computational graphs are written using pure Theano, and based on the implementations in [dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial).

We refer each source-target pair a computational graph, since we build an actual separate computational graph for each of them, where some of the parameters in these computational graphs are shared with other computational graphs.<br>

In order to train multiple computational graphs, we need multiple data-streams, and a scheduler over them. This part is handled by Fuel and custom streams, along with development and test decoding streams.<br>

Given the computational graphs and their corresponding data streams, training the parameters in the computational graphs is carried out by adapted training loop from Blocks.<br>

Finally, this codebase is a refined combination of multiple codebases. The layer structure and handling of parameters are somehow similar to dl4mt-tutorial. The class hierarchy and experiment configuration resembles a pruned version of [GroundHog](https://github.com/lisa-groundhog/GroundHog) and main-loop and extensions are quite similar to
[blocks-examples](https://github.com/mila-udem/blocks-examples/tree/master/machine_translation).

During the development of this codebase, we tried to be pragmatic and inherit the lessons learned from other NMT implementations, hope we picked the best parts not the worst :relieved:

Preparing Text Corpora:
-----------------------
The original text corpora could be downloaded from [here](http://www.statmt.org/wmt15/translation-task.html).<br>

In this repo, we do not handle downloading the data and tokenizing it. Please follow the steps described in dl4mt-tutorial for downloading and tokenization of the data. Once you've downloaded and tokenized the data, you can use `scripts/encode_with_bpe_parallel.sh` and `scripts/encode_with_bpe_joint.sh` to
use sub-word units as input and output tokens (check scripts for details).
