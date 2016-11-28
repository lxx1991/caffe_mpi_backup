#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_Final_VOC_val/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_Final_VOC_val/IRes_VOC_solver_finetune.prototxt \
    --snapshot=examples/IRes_Final_VOC_val/model/unary_mat_iter_13500.solverstate
