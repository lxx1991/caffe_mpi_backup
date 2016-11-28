#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_Final_VOC/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_Final_VOC/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_DSN_DROP_VOC/95_98_G/model_back/best.caffemodel
