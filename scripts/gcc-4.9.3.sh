#!/bin/bash

rm -rf build-gnu-4.9.3-release 2>/dev/null
mkdir build-gnu-4.9.3-release && cd build-gnu-4.9.3-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_C_COMPILER=/usr/apps/gnu/4.9.3/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/apps/gnu/4.9.3/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-gnu-4.9.3-release \
  "$@" \
  ${RAJA_DIR}
