#!/bin/bash

rm -rf build-clang-cuda-release 2>/dev/null
mkdir build-clang-cuda-release && cd build-clang-cuda-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/clang_cuda.cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-8.0 \
  -DCMAKE_INSTALL_PREFIX=../install-clang-cuda-release \
  "$@" \
  ${RAJA_DIR}
