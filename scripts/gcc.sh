#!/bin/bash

rm -f build-gnu-release 2>/dev/null
mkdir build-gnu-release && cd build-gnu-release

cmake \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-gnu-release \
  "$@" \
  ../../
