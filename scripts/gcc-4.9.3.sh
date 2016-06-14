#!/bin/bash

rm -rf build-gnu-4.9.3-release 2>/dev/null
mkdir build-gnu-4.9.3-release && cd build-gnu-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gnu_4_9_3.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-gnu-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
