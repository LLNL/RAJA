#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
sys_type=${SYS_TYPE:-""}
compiler=${COMPILER:-""}
hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}

# Dependencies
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    python scripts/uberenv/uberenv.py --spec=${spec}

fi

# Host config file
if [[ -z ${hostconfig} ]]
then
    # Attempt to retrieve host-config from env. We need sys_type and compiler.
    if [[ -z ${sys_type} ]]
    then
        echo "SYS_TYPE is undefined, aborting..."
        exit 1
    fi
    if [[ -z ${compiler} ]]
    then
        echo "COMPILER is undefined, aborting..."
        exit 1
    fi
    hostconfig="${hostname//[0-9]/}-${sys_type}-${compiler}.cmake"

    # First try with where uberenv generates host-configs.
    hostconfig_path="${project_dir}/${hostconfig}"
    if [[ ! -f ${hostconfig_path} ]]
    then
        echo "File not found: ${hostconfig_path}"
        echo "Spack generated host-config not found, trying with predefined"
    fi
    # Otherwise look into project predefined host-configs.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
    if [[ ! -f ${hostconfig_path} ]]
    then
        echo "File not found: ${hostconfig_path}"
        echo "Predefined host-config not found, aborting"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

# Build Directory
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

build_dir="${build_root}/build_${hostconfig//.cmake/}"
install_dir="${build_root}/install_${hostconfig//.cmake/}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Host-config: ${hostconfig_path}"
echo "~~~~~ Build Dir:   ${build_dir}"
echo "~~~~~ Project Dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building RAJA"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    module load cmake/3.14.5

    cmake \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
      -C ${hostconfig_path} \
      -DENABLE_OPENMP=${ENABLE_OPENMP:-On} \
      -DCMAKE_INSTALL_PREFIX=${install_dir} \
      ${project_dir}
    cmake --build . -j
fi

# Test
if [[ "${option}" != "--build-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing RAJA"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "ERROR: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    ctest --output-on-failure -T test 2>&1 | tee tests_output.txt

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    cp Testing/*/Test.xml ${project_dir}

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi
fi
