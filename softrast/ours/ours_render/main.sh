#!/usr/bin/env bash

PYTHONPATH=../../../FreeTensor/python:../../../FreeTensor/build:$PYTHONPATH srun -N 1 -p v100 --exclusive python3 freetensor_render_backward_full.py
