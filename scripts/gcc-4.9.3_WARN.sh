#!/bin/bash

rm -rf build-gcc-4.9.3-debug_WARN 2>/dev/null
mkdir build-gcc-4.9.3-debug_WARN && cd build-gcc-4.9.3-debug_WARN

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/gcc_4_9_3.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DRAJA_ENABLE_APPLICATIONS=On \
  -DRAJA_ENABLE_WARNINGS=On \
  "$@" \
  ${RAJA_DIR}
