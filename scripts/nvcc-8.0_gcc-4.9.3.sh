#!/bin/bash

##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For additional details and restrictions, please see RAJA/LICENSE.txt
##

rm -rf build-nvcc-8.0_gcc-4.9.3 2>/dev/null
mkdir build-nvcc-8.0_gcc-4.9.3 && cd build-nvcc-8.0_gcc-4.9.3

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/nvcc.cmake \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-8.0 \
  -DCMAKE_INSTALL_PREFIX=../install-nvcc-8.0_gcc-4.9.3 \
  "$@" \
  ${RAJA_DIR}
