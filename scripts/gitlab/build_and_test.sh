#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -o errexit
set -o nounset

# TODO : ill-named environment variable
echo "--- Configuration to match :"
echo "* ${COMPILER}"

# FIRST STEP :
# Find raja configuration matching the toolchain
available_confs="$(ls host-configs/${SYS_TYPE}/ | grep "\.cmake$")"
echo "--- Available configurations"
echo "${available_confs}"

match_count=0
configuration=""

# Translate file names into pattern to match the configuration
echo "--- Patterns"
for conf in ${available_confs}
do
    pattern="${conf//X/.*}"
    pattern="${pattern/.cmake/}"
    echo "${pattern}"

    if [[ "${COMPILER}" =~ ^${pattern}$ ]]
    then
        (( ++match_count ))
        configuration="${conf}"
        echo "-> Found Project Conf : ${configuration}"
    fi
done

if (( match_count > 1 )) || (( match_count == 0 ))
then
    echo "ERROR : none or multiple match(s) ..."
    exit 1
fi

# SECOND STEP :
# Build
build_suffix="${SYS_TYPE}_${COMPILER}"
build_dir="build_${build_suffix}"
echo "--- Build (${build_dir})"

rm -rf ${build_dir} 2>/dev/null
mkdir ${build_dir} && cd ${build_dir}

install_dir="../install_${build_suffix}"
compiler_conf="../.gitlab/conf/host-configs/${SYS_TYPE}/${COMPILER}.cmake"
raja_conf="../host-configs/${SYS_TYPE}/${configuration}"

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${compiler_conf} \
  -C ${raja_conf} \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=${install_dir} \
  .. 
make -j 8
ctest -V -T test
