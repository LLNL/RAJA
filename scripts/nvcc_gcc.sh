#!/bin/bash

rm -rf build-nvcc_gcc-release 2>/dev/null
mkdir build-nvcc_gcc-release && cd build-nvcc_gcc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAJA_ENABLE_CUDA=On \
  -DCMAKE_INSTALL_PREFIX=../install-nvcc_gcc-release \
  "$@" \
  ${RAJA_DIR}
