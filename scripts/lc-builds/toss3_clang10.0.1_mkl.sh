#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=lc_toss3-clang-10.0.1_mkl

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-10.0.1/bin/clang++ \
  -C ../host-configs/lc-builds/toss3/clang_X.cmake \
  -DENABLE_OPENMP=On \
	-DENABLE_BLAS=On \
	-DBLA_VENDOR=Intel10_64lp_seq \
	-Dcblas_DIR=/usr/gapps/bdiv/toss_3_x86_64_ib/clang-10-mvapich2-2.3/lapack/3.9.0/lib64/cmake/cblas-3.9.0 \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  .. 
  
