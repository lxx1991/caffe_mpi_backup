#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_VOC/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_VOC/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_VOC/model.caffemodel
