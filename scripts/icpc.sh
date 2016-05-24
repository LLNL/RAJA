#!/bin/bash

rm -f build-icpc-release 2>/dev/null
mkdir build-icpc-release && cd build-icpc-release

cmake \
  -DCMAKE_C_COMPILER=icc \
  -DCMAKE_CXX_COMPILER=icpc \
  -DRAJA_ENABLE_CILK=On \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-release \
  "$@" \
  ../../
