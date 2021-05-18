#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss3_icpc.sh 19.1.0"
  exit
fi

COMP_VER=$1
COMP_MAJOR_VER=${COMP_VER:0:2}
GCC_HEADER_VER=7
USE_TBB=On

if [ ${COMP_MAJOR_VER} -gt 18 ]
then
  GCC_HEADER_VER=8 
fi

if [ ${COMP_MAJOR_VER} -lt 18 ]
then 
  USE_TBB=Off
fi

BUILD_SUFFIX=lc_toss3-icpc-${COMP_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

##
# CMake option -DENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile 
# times at a potential cost of slower 'forall' execution.
##

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icc \
  -DBLT_CXX_STD=c++11 \
  -C ../host-configs/lc-builds/toss3/icpc_X_gcc${GCC_HEADER_VER}headers.cmake \
  -DENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_OPENMP=On \
  -DENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "**********************************************************************"
echo "cd into directory ${BUILD_SUFFIX} and run make to build RAJA"
echo "**********************************************************************"
