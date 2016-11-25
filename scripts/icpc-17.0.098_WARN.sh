#!/bin/bash

rm -rf build-icpc-17.0.098-debug_WARN 2>/dev/null
mkdir build-icpc-17.0.098-debug_WARN && cd build-icpc-17.0.098-debug_WARN

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/icpc_17_0_098.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DRAJA_ENABLE_APPLICATIONS=On \
  -DRAJA_ENABLE_WARNINGS=On \
  "$@" \
  ${RAJA_DIR}
