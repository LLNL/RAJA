#!/bin/bash

rm -rf build-nvcc-7.5_gcc-4.9.3-debug_WARN 2>/dev/null
mkdir build-nvcc-7.5_gcc-4.9.3-debug_WARN && cd build-nvcc-7.5_gcc-4.9.3-debug_WARN

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/nvcc.cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-7.5 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DRAJA_ENABLE_APPLICATIONS=On \
  "$@" \
  ${RAJA_DIR}
