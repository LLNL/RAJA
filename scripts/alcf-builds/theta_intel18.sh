#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

##
## The following command will request a node in the debug partition:
##
## qsub -I -n 1 -A your_account -t 60 -q debug-cache-quad
##
## Example of running an executable:
##
## "aprun ./main"
##
## Execute these commands before running this script to
## set up your build environment:
##
##  module load intel/18.0.0.128
##  module load cmake/3.9.1
##

BUILD_SUFFIX=alcf-theta-intel18.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/alcf-builds/theta_intel18_0.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
