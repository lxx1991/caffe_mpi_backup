#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_Final_VOC/LOG \
    /usr/local/openmpi/bin/mpirun -np 4 \
    build/install/bin/caffe train \
    --solver=examples/IRes_Final_VOC/IRes_VOC_solver_finetune.prototxt \
    --snapshot=examples/IRes_Final_VOC/model/finetune_iter_20000.solverstate
