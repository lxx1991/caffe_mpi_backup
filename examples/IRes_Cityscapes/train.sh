#!/usr/bin/env sh

GOOGLE_LOG_DIR=examples/IRes_Cityscapes/LOG \
    /usr/local/openmpi/bin/mpirun -np 8 \
	    build/install/bin/caffe train \
    --solver=examples/IRes_Cityscapes/IRes_Cityscapes_solver.prototxt \
    --weights=examples/IRes_Cityscapes/model.caffemodel
