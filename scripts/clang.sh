#!/bin/bash

rm -rf build-clang-release 2>/dev/null
mkdir build-clang-release && cd build-clang-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/clang.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-clang-release \
  "$@" \
  ${RAJA_DIR}
