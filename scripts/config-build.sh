#!/usr/bin/env bash

#------------------------------------------------------------------------------
#clean up existing build + install directories
#------------------------------------------------------------------------------
rm -rf build-debug install-debug

#------------------------------------------------------------------------------
# create a new build directory
#------------------------------------------------------------------------------
mkdir build-debug
mkdir install-debug
cd build-debug

echo "Changing to build directory..."

#------------------------------------------------------------------------------
# setup desired basic cmake options
#------------------------------------------------------------------------------
export CMAKE_OPTS=" -DCMAKE_BUILD_TYPE=Debug"
export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=../install-debug"

#------------------------------------------------------------------------------
# include an initial cmake settings file if appropriate
#------------------------------------------------------------------------------

# first look for a specific config for this machine
export HOST_CONFIG=../host-configs/other/`hostname`.cmake
echo "Looking for host-config file: $HOST_CONFIG"
if [[ -e  "$HOST_CONFIG" ]]; then
    echo "FOUND: $HOST_CONFIG"
    export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
# then check for a sys-type based config
elif [[ "$SYS_TYPE" != "" ]]; then
    export HOST_CONFIG=../host-configs/$SYS_TYPE.cmake
    echo "Looking for SYS_TYPE based host-config file: $HOST_CONFIG"
    if [[ -e  "$HOST_CONFIG" ]]; then
        echo "FOUND: $HOST_CONFIG"
        export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
    fi
else 
    export HOST_CONFIG=../host-configs/other/`uname`.cmake
    echo "Looking for uname based host-config file: $HOST_CONFIG"
    if [[ -e  "$HOST_CONFIG" ]]; then
        echo "FOUND: $HOST_CONFIG"
        export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
    fi
fi

#------------------------------------------------------------------------------
# parse other command line arguments
#------------------------------------------------------------------------------
for arg in "$@"
do
    arg="$1"
    echo "ARGUMENT: $arg"
    case $arg in
        *)
        # unknown option, skip
        shift
        ;;
    esac
done


#------------------------------------------------------------------------------
# run cmake to configure
#------------------------------------------------------------------------------
echo "Executing cmake line: cmake $CMAKE_OPTS ../.."
eval "cmake  $CMAKE_OPTS ../.."

# return to the starting dir
cd ../
