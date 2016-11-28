#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_Final_Cityscapes/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_Final_Cityscapes/IRes_Cityscapes_solver.prototxt \
    --snapshot=examples/IRes_Final_Cityscapes/unary_mat_iter_10000.solverstate
