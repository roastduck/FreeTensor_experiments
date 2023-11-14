#!/usr/bin/env bash
set -e

#PYTHONPATH=../../../FreeTensor/python:../../../FreeTensor/build:$PYTHONPATH srun -N 1 --pty --exclusive python3 freetensor_render_backward_full.py
python3 freetensor_render_backward_full.py

python3 compare.py
