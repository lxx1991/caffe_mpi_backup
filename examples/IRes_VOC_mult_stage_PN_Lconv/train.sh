#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_VOC_mult_stage_PN_Lconv/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_VOC_mult_stage_PN_Lconv/IRes_VOC_solver.prototxt \
    --weights=examples/IRes_VOC_mult_stage_PN/model/unary_mat_iter_10000.caffemodel
