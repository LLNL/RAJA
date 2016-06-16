##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## All rights reserved.
##
## For release details and restrictions, please see raja/README-license.txt
##

set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/collab/usr/global/tools/intel/chaos_5_x86_64_ib/ipsxe-2017.0-beta/bin/icpc" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -xMIC-AVX512 -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -xMIC-AVX512 -inline-forceinline -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

set(RAJA_ENABLE_OPENMP On CACHE BOOL "")
set(RAJA_ENABLE_CILK On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
