#!/bin/bash

rm -rf build-clang-3.7.0-release 2>/dev/null
mkdir build-clang-3.7.0-release && cd build-clang-3.7.0-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_C_COMPILER=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.7.0/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.7.0/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAJA_ENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install-clang-3.7.0-release \
  "$@" \
  ${RAJA_DIR}
