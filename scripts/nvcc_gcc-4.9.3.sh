#!/bin/bash

rm -rf build-nvcc_gcc-4.9.3-release 2>/dev/null
mkdir build-nvcc_gcc-4.9.3-release && cd build-nvcc_gcc-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/nvcc.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-nvcc_gcc-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
