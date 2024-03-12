#! /bin/sh -e
set PYSEN_CONFIG "./pysen.toml"


function setup-conda() {
    set CURRENT_DIR $(pwd)
    cd $(git rev-parse --show-toplevel)
    conda create -n gzcv python=3.9 -y
    conda activate gzcv
    pip install --upgrade pip
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -e .
    cd $CURRENT_DIR
}

function build-sif() {
    set CURRENT_DIR $(pwd)
    cd $(git rev-parse --show-toplevel)
    sudo singularity build gzcv.sif singularity.def
    cd $CURRENT_DIR
}

function lint () {
    set CURRENT_DIR $(pwd)
    cd $(git rev-parse --show-toplevel)
    pysen --config $PYSEN_CONFIG run lint
    pysen --config $PYSEN_CONFIG run format
    cd $CURRENT_DIR
}
