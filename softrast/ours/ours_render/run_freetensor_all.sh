#!/usr/bin/bash

set -ex

export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=true

python3 ./freetensor_render_backward_full.py 000 16  >./log/freetensor_render_000_16.txt 2>&1
python3 ./freetensor_render_backward_full.py 000 32  >./log/freetensor_render_000_32.txt 2>&1
python3 ./freetensor_render_backward_full.py 000 64  >./log/freetensor_render_000_64.txt 2>&1
python3 ./freetensor_render_backward_full.py 000 128 >./log/freetensor_render_000_128.txt 2>&1
python3 ./freetensor_render_backward_full.py 000 256 >./log/freetensor_render_000_256.txt 2>&1

PAPER_SERIAL=1 python3 ./freetensor_render_backward_full.py 000 16  >./log/freetensor_render_serial_000_16.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_backward_full.py 000 32  >./log/freetensor_render_serial_000_32.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_backward_full.py 000 64  >./log/freetensor_render_serial_000_64.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_backward_full.py 000 128 >./log/freetensor_render_serial_000_128.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_backward_full.py 000 256 >./log/freetensor_render_serial_000_256.txt 2>&1

python3 ./freetensor_render_objective_only.py 000 16  >./log/freetensor_render_objective_000_16.txt 2>&1
python3 ./freetensor_render_objective_only.py 000 32  >./log/freetensor_render_objective_000_32.txt 2>&1
python3 ./freetensor_render_objective_only.py 000 64  >./log/freetensor_render_objective_000_64.txt 2>&1
python3 ./freetensor_render_objective_only.py 000 128 >./log/freetensor_render_objective_000_128.txt 2>&1
python3 ./freetensor_render_objective_only.py 000 256 >./log/freetensor_render_objective_000_256.txt 2>&1

PAPER_SERIAL=1 python3 ./freetensor_render_objective_only.py 000 16  >./log/freetensor_render_serial_objective_000_16.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_objective_only.py 000 32  >./log/freetensor_render_serial_objective_000_32.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_objective_only.py 000 64  >./log/freetensor_render_serial_objective_000_64.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_objective_only.py 000 128 >./log/freetensor_render_serial_objective_000_128.txt 2>&1
PAPER_SERIAL=1 python3 ./freetensor_render_objective_only.py 000 256 >./log/freetensor_render_serial_objective_000_256.txt 2>&1
