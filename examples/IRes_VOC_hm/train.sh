#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_VOC_hm/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_VOC_hm/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_VOC_hm/model.caffemodel
