#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_DSN_DROP_VOC/95_1/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_DSN_DROP_VOC/95_1/IRes_VOC_solver2.prototxt \
    --snapshot=examples/IRes_DSN_DROP_VOC/95_1/model/unary_mat_iter_15000.solverstate
