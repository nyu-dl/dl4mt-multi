#!/bin/bash
##############################################################################
# Sets the development environment for dl4mt-multi.
#
# We use a specific version of blocks (quite old), a specific version of fuel
# (again quite old), and picklable_itertools (well, probably old).
#
# You may also need a few more python packages, which are not installed here:
#   - numpy (pip install numpy --user)
#   - h5 (pip install h5py --user)
#   - progressbar (pip install progressbar --user) 
#   - six (pip install six --user)
##############################################################################

CWD=$PWD

# we will put all the repos here
CODES_DIR=${HOME}/codes

if [ ! -d "$CODES_DIR" ]; then
    mkdir -p $CODES_DIR 
fi 

cd $CODES_DIR

# clone blocks
git clone -b multi_enc_05 https://github.com/orhanf/blocks

# clone fuel
git clone -b nmt https://github.com/orhanf/fuel

# clone picklable-itertools
git clone https://github.com/orhanf/picklable_itertools

# checkout a specific commit of blocks
cd blocks
git checkout 117d3154b32aaece5262de66922216202bfc7f63
cd $CWD

echo ""
echo "Please add the following paths to your PYTHONPATH"
echo "\${HOME}/codes/blocks"
echo "\${HOME}/codes/fuel"
echo "\${HOME}/codes/picklable_itertools"
echo ""
echo "or simply put these in your .bashrc and then source it"
echo "export PYTHONPATH=\${HOME}/codes/blocks:\$PYTHONPATH"
echo "export PYTHONPATH=\${HOME}/codes/fuel:\$PYTHONPATH"
echo "export PYTHONPATH=\${HOME}/codes/picklable_itertools:\$PYTHONPATH"
echo ""
echo "You may also need to setup fuel, blocks and picklable itertools manually,"
echo "by going into the cloned directories and calling:"
echo "  python ./setup.py develop --user"
echo ""
