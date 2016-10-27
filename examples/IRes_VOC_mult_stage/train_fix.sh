#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_VOC_mult_stage/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_VOC_mult_stage/IRes_VOC_solver_fix.prototxt \
    --weights=examples/IRes_VOC_mult_stage/IRes_VOC_20000.caffemodel
