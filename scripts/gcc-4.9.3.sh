#!/bin/bash

rm -rf build-gcc-4.9.3-release 2>/dev/null
mkdir build-gcc-4.9.3-release && cd build-gcc-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gcc_4_9_3.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-gcc-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
