#!/bin/bash

rm -rf build-clang-3.9.0-debug_WARN 2>/dev/null
mkdir build-clang-3.9.0-debug_WARN && cd build-clang-3.9.0-debug_WARN

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/clang_3_9_0.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DRAJA_ENABLE_APPLICATIONS=On \
  -DRAJA_ENABLE_WARNINGS=On \
  "$@" \
  ${RAJA_DIR}
