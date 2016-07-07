#!/bin/bash

rm -rf build-nvcc-8.0_gcc-4.9.3-release 2>/dev/null
mkdir build-nvcc-8.0_gcc-4.9.3-release && cd build-nvcc-8.0_gcc-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/nvcc.cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-8.0 \
  -DCMAKE_INSTALL_PREFIX=../install-nvcc-8.0_gcc-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
