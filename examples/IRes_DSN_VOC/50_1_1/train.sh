#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_DSN_VOC/50_1_1/LOG \
    /usr/local/openmpi/bin/mpirun -np 8 \
    build/install/bin/caffe train \
    --solver=examples/IRes_DSN_VOC/50_1_1/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_VOC/model/best.caffemodel
