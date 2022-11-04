#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss4_icpc.sh 2021.6.0"
  exit
fi

COMP_VER=$1
shift 1

USE_TBB=On

BUILD_SUFFIX=lc_toss4-icpc-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.21.1

##
# CMake option -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile
# times at a potential cost of slower 'forall' execution.
##

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel-classic/intel-classic-${COMP_VER}/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel-classic/intel-classic-${COMP_VER}/bin/icc \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/toss4/icpc_X.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "  Please note that you may need to add some intel openmp libraries to your"
echo "  LD_LIBRARY_PATH to run with openmp."
echo
echo "    LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/tce/packages/intel-classic-tce/intel-classic-2021.6.0/compiler/lib/intel64_lin"
echo
echo "***********************************************************************"
