#!/bin/bash

rm -rf build-bgq_gnu-4.7.2-release 2>/dev/null
mkdir build-bgq_gnu-4.7.2-release && cd build-bgq_gnu-4.7.2-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/bgqos/gnu_4_7_2.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-bgq_gnu-4.7.2-release \
  "$@" \
  ${RAJA_DIR}
