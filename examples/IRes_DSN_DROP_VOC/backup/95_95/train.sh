#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_DSN_DROP_VOC/95_95/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_DSN_DROP_VOC/95_95/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_DSN_DROP_VOC/95_95/IRes_VOC_mult_stage_10000.caffemodel
