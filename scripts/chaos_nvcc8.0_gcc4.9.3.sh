#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-xxxxxx
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For more information, see the file LICENSE in the top-level directory.
##

rm -rf build_chaos-nvcc8.0_gcc4.9.3 2>/dev/null
mkdir build_chaos-nvcc8.0_gcc4.9.3 && cd build_chaos-nvcc8.0_gcc4.9.3
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/nvcc_gcc4_9_3.cmake \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-8.0 \
  -DRAJA_ENABLE_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-nvcc8.0_gcc4.9.3 \
  "$@" \
  ${RAJA_DIR}
