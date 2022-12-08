#!/usr/bin/env bash

# Initialize modules for users not using bash as a default shell
if test -e /usr/share/lmod/lmod/init/bash
then
  . /usr/share/lmod/lmod/init/bash
fi

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and RAJA
# project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
truehostname=${hostname//[0-9]/}
project_dir="$(pwd)"

build_root=${BUILD_ROOT:-""}
hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}
job_unique_id=${CI_JOB_ID:-""}

prefix=""

if [[ -d /dev/shm ]]
then
    prefix="/dev/shm/${hostname}"
    if [[ -z ${job_unique_id} ]]; then
      job_unique_id=manual_job_$(date +%s)
      while [[ -d ${prefix}-${job_unique_id} ]] ; do
          sleep 1
          job_unique_id=manual_job_$(date +%s)
      done
    fi

    prefix="${prefix}-${job_unique_id}"
    mkdir -p ${prefix}
fi

# Dependencies
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test started"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
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

    prefix_opt=""

    if [[ -d /dev/shm ]]
    then
        prefix_opt="--prefix=${prefix}"

        # We force Spack to put all generated files (cache and configuration of
        # all sorts) in a unique location so that there can be no collision
        # with existing or concurrent Spack.
        spack_user_cache="${prefix}/spack-user-cache"
        export SPACK_DISABLE_LOCAL_CONFIG=""
        export SPACK_USER_CACHE_PATH="${spack_user_cache}"
        mkdir -p ${spack_user_cache}
    fi

    ./scripts/uberenv/uberenv.py --spec="${spec}" ${prefix_opt}

fi
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~ Dependencies Built"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/*.cmake"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/*.cmake"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/host-configs/${hostconfig}"
fi

hostconfig=$(basename ${hostconfig_path})

# Build Directory
if [[ -z ${build_root} ]]
then
    if [[ -d /dev/shm ]]
    then
        build_root="${prefix}"
    else
        build_root="$(pwd)"
    fi
else
    build_root="${build_root}"
fi

build_dir="${build_root}/build_${hostconfig//.cmake/}"
install_dir="${build_root}/install_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Host-config: ${hostconfig_path}"
    echo "~~~~~ Build Dir:   ${build_dir}"
    echo "~~~~~ Project Dir: ${project_dir}"
    echo "~~~~~ Install Dir: ${install_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building RAJA"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Map CPU core allocations
    declare -A core_counts=(["lassen"]=40 ["ruby"]=28 ["corona"]=32 ["rzansel"]=48)

    # If building, then delete everything first
    # NOTE: 'cmake --build . -j core_counts' attempts to reduce individual build resources.
    #       If core_counts does not contain hostname, then will default to '-j ', which should
    #       use max cores.
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    date

    if [[ "${truehostname}" == "corona" ]]
    then
        module unload rocm
    fi
    $cmake_exe \
      -C ${hostconfig_path} \
      -DCMAKE_INSTALL_PREFIX=${install_dir} \
      ${project_dir}
    if ! $cmake_exe --build . -j ${core_counts[$truehostname]}
    then
        echo "ERROR: compilation failed, building with verbose output..."
        $cmake_exe --build . --verbose -j 1
    else
        make install
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ RAJA Built"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    date
fi

# Test
if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_TESTS.*ON" ${hostconfig_path}
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing RAJA"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "ERROR: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    # If HIP enabled
    if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_HIP.*ON" ${hostconfig_path}
    then # don't run the tests that are known to fail
        date
        ctest --output-on-failure -T test 2>&1 -E Known-Hip-Failure | tee tests_output.txt
        date
    else #run all tests like normal
        date
        ctest --output-on-failure -T test 2>&1 | tee tests_output.txt
        date
    fi

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    echo "Copying Testing xml reports for export"
    tree Testing
    xsltproc -o junit.xml ${project_dir}/blt/tests/ctest-to-junit.xsl Testing/*/Test.xml
    mv junit.xml ${project_dir}/junit.xml

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi

    if grep -q -i "ENABLE_HIP.*ON" ${hostconfig_path}
    then
        echo "WARNING: not testing install with HIP"
    else
        if [[ ! -d ${install_dir} ]]
        then
            echo "ERROR: install directory not found : ${install_dir}" && exit 1
        fi

        cd ${install_dir}/examples/RAJA/using-with-cmake
        mkdir build && cd build
        if ! $cmake_exe -C ../host-config.cmake ..; then
        echo "ERROR: running $cmake_exe for using-with-cmake test" && exit 1
        fi

        if ! make; then
        echo "ERROR: running make for using-with-cmake test" && exit 1
        fi
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ RAJA Tests Complete"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    date

    # echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # echo "~~~~~ CLEAN UP"
    # echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # make clean
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test completed"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date
