#!/bin/bash

rm -rf build-bgq_clang-3.9.0-release 2>/dev/null
mkdir build-bgq_clang-3.9.0-release && cd build-bgq_clang-3.9.0-release
. /usr/local/tools/dotkit/init.sh && use cmake-3.1.2

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/bgqos/clang_3_9_0.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-bgq_clang-3.9.0-release \
  "$@" \
  ${RAJA_DIR}
