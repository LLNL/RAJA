#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=ubuntu-hipcc

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ../host-configs/ubuntu-builds/hip.cmake \
  ..
