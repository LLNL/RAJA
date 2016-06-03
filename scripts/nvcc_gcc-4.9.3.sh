#!/bin/bash

rm -rf build-nvcc_gcc-4.9.3-release 2>/dev/null
mkdir build-nvcc_gcc-4.9.3-release && cd build-nvcc_gcc-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_C_COMPILER=/usr/apps/gnu/4.9.3/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/apps/gnu/4.9.3/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAJA_ENABLE_CUDA=On \
  -DCMAKE_INSTALL_PREFIX=../install-nvcc_gcc-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
