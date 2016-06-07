#!/bin/bash

rm -rf build-clang-release 2>/dev/null
mkdir build-clang-release && cd build-clang-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAJA_ENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install-clang-release \
  "$@" \
  ${RAJA_DIR}
