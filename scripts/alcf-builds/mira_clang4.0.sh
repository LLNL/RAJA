#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

##
## Execute these commands before running this script to
## set up your build environment:
##
##  > soft add +cmake-3.3.0
##  > soft add +mpiwrapper-bgclang-mpi3
##

BUILD_SUFFIX=alcf-mira-clang4.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/alcf-builds/mira_clang4_0.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
