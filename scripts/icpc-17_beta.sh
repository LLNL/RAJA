#!/bin/bash

rm -rf build-icpc-17_beta-release 2>/dev/null
mkdir build-icpc-17_beta-release && cd build-icpc-17_beta-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/icpc-17_beta.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-1cpc-17_beta-release \
  "$@" \
  ${RAJA_DIR}
