##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## All rights reserved.
##
## For release details and restrictions, please see raja/README-license.txt
##

set(RAJA_COMPILER "RAJA_COMPILER_XLC12" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/local/tools/compilers/ibm/mpixlcxx_r-lompbeta2-fastmpi" CACHE PATH "")

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=auto:level=10 -qsmp=omp" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -qarch=qp -qhot=novector -qsimd=auto -qlanglvl=extended0x -qnostrict -qinline=auto:level=10 -qsmp=omp" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -qarch=qp -qlanglvl=extended0x -qsmp=omp" CACHE STRING "")
endif()

set(RAJA_USE_OPENMP On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
