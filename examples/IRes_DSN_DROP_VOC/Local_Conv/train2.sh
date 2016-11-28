#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_DSN_DROP_VOC/Local_Conv/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_DSN_DROP_VOC/Local_Conv/IRes_VOC_solver2.prototxt \
    --weights=examples/IRes_DSN_DROP_VOC/best.caffemodel
