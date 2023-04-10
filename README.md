# rustlets
A wavelet library written in Rust, with bindings to python.


## steps to build

Make sure you have rust and python installed, check rustc --version

Setup virtual environment to your liking, make sure to export PYTHON_CONFIGURE_OPTS="--enable-shared"

Install maturin python package into virtual environment. Install numpy as well

For example, with pip, do 
pip install maturin


You will also need matplotlib, scipy, numpy to run demos


To build, run maturin develop --release

## usage
see cwtfuncs.py

## TODO
Move Array types to ArcArray for cleanliness
