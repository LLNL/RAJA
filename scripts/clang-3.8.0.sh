#!/bin/bash

rm -rf build-clang-3.8.0-release 2>/dev/null
mkdir build-clang-3.8.0-release && cd build-clang-3.8.0-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/clang_3_8_0.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-clang-3.8.0-release \
  "$@" \
  ${RAJA_DIR}
