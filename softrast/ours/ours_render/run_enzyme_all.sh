#!/usr/bin/bash

set -ex

export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=true

./enzyme_render 000 16  >./log/enzyme_render_000_16.txt 2>&1
./enzyme_render 000 32  >./log/enzyme_render_000_32.txt 2>&1
./enzyme_render 000 64  >./log/enzyme_render_000_64.txt 2>&1
./enzyme_render 000 128 >./log/enzyme_render_000_128.txt 2>&1
./enzyme_render 000 256 >./log/enzyme_render_000_256.txt 2>&1

./enzyme_render_serial 000 16  >./log/enzyme_render_serial_000_16.txt 2>&1
./enzyme_render_serial 000 32  >./log/enzyme_render_serial_000_32.txt 2>&1
./enzyme_render_serial 000 64  >./log/enzyme_render_serial_000_64.txt 2>&1
./enzyme_render_serial 000 128 >./log/enzyme_render_serial_000_128.txt 2>&1
./enzyme_render_serial 000 256 >./log/enzyme_render_serial_000_256.txt 2>&1

./enzyme_render_objective 000 16  >./log/enzyme_render_objective_000_16.txt 2>&1
./enzyme_render_objective 000 32  >./log/enzyme_render_objective_000_32.txt 2>&1
./enzyme_render_objective 000 64  >./log/enzyme_render_objective_000_64.txt 2>&1
./enzyme_render_objective 000 128 >./log/enzyme_render_objective_000_128.txt 2>&1
./enzyme_render_objective 000 256 >./log/enzyme_render_objective_000_256.txt 2>&1

./enzyme_render_serial_objective 000 16  >./log/enzyme_render_serial_objective_000_16.txt 2>&1
./enzyme_render_serial_objective 000 32  >./log/enzyme_render_serial_objective_000_32.txt 2>&1
./enzyme_render_serial_objective 000 64  >./log/enzyme_render_serial_objective_000_64.txt 2>&1
./enzyme_render_serial_objective 000 128 >./log/enzyme_render_serial_objective_000_128.txt 2>&1
./enzyme_render_serial_objective 000 256 >./log/enzyme_render_serial_objective_000_256.txt 2>&1
