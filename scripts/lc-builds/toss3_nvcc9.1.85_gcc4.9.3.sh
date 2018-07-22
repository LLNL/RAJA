#!/bin/bash

##
## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

rm -rf build_toss3-nvcc9.1.85_gcc4.9.3 2>/dev/null
mkdir build_toss3-nvcc9.1.85_gcc4.9.3 && cd build_toss3-nvcc9.1.85_gcc4.9.3

module load cmake/3.5.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/toss3/nvcc_gcc4_9_3.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-9.1.85 \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_toss3-nvcc9.1.85_gcc4.9.3 \
  "$@" \
  ..
