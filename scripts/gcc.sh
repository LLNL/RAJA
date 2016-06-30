#!/bin/bash

rm -rf build-gcc-release 2>/dev/null
mkdir build-gcc-release && cd build-gcc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gcc.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-gcc-release \
  "$@" \
  ${RAJA_DIR}
