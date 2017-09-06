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

rm -rf build_chaos-clang_cuda8.0 2>/dev/null
mkdir build_chaos-clang_cuda8.0 && cd build_chaos-clang_cuda8.0
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.1

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/chaos/clang_cuda.cmake \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_CUDA=On \
  -DRAJA_ENABLE_CLANG_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-8.0 \
  -DRAJA_ENABLE_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_chaos-clang_cuda8.0 \
   "$@" \
   ${RAJA_DIR}
