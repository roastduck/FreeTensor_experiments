#!/usr/bin/bash

set -ex

export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=true

cd tapenade/build

cmake .. -DIMAGE_SIZE=16  && make -B && ./omp 000 > ../../log/tapenade_render_serial_000_16.txt  2>&1
cmake .. -DIMAGE_SIZE=32  && make -B && ./omp 000 > ../../log/tapenade_render_serial_000_32.txt  2>&1
cmake .. -DIMAGE_SIZE=64  && make -B && ./omp 000 > ../../log/tapenade_render_serial_000_64.txt  2>&1
cmake .. -DIMAGE_SIZE=128 && make -B && ./omp 000 > ../../log/tapenade_render_serial_000_128.txt 2>&1
cmake .. -DIMAGE_SIZE=256 && make -B && ./omp 000 > ../../log/tapenade_render_serial_000_256.txt 2>&1

cd ../..

cd tapenade_objective/build

cmake .. -DIMAGE_SIZE=16  && make -B && ./omp 000 > ../../log/tapenade_render_objective_000_16.txt  2>&1
cmake .. -DIMAGE_SIZE=32  && make -B && ./omp 000 > ../../log/tapenade_render_objective_000_32.txt  2>&1
cmake .. -DIMAGE_SIZE=64  && make -B && ./omp 000 > ../../log/tapenade_render_objective_000_64.txt  2>&1
cmake .. -DIMAGE_SIZE=128 && make -B && ./omp 000 > ../../log/tapenade_render_objective_000_128.txt 2>&1
cmake .. -DIMAGE_SIZE=256 && make -B && ./omp 000 > ../../log/tapenade_render_objective_000_256.txt 2>&1

cd ../..

cd tapenade_serial_objective/build

cmake .. -DIMAGE_SIZE=16  && make -B && ./serial 000 > ../../log/tapenade_render_serial_objective_000_16.txt  2>&1
cmake .. -DIMAGE_SIZE=32  && make -B && ./serial 000 > ../../log/tapenade_render_serial_objective_000_32.txt  2>&1
cmake .. -DIMAGE_SIZE=64  && make -B && ./serial 000 > ../../log/tapenade_render_serial_objective_000_64.txt  2>&1
cmake .. -DIMAGE_SIZE=128 && make -B && ./serial 000 > ../../log/tapenade_render_serial_objective_000_128.txt 2>&1
cmake .. -DIMAGE_SIZE=256 && make -B && ./serial 000 > ../../log/tapenade_render_serial_objective_000_256.txt 2>&1

cd ../..
