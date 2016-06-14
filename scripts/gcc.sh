#!/bin/bash

rm -rf build-gnu-release 2>/dev/null
mkdir build-gnu-release && cd build-gnu-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gnu.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-gnu-release \
  "$@" \
  ${RAJA_DIR}
