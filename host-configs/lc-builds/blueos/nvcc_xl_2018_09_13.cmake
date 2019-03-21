##
## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_XLC" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/xl/xl-beta-2018.09.13/bin/xlc++_r" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/xl/xl-beta-2018.09.13/bin/xlC_r" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3  " CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g9 " CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -qsmp=omp:noopt " CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,muldefs" CACHE STRING "")

set(CUDA_COMMON_OPT_FLAGS -restrict; -arch sm_60; -std c++11;  --generate-line-info; --expt-extended-lambda; --expt-relaxed-constexpr)
set(CUDA_COMMON_DEBUG_FLAGS -restrict; -arch sm_60; -std c++11; --expt-extended-lambda; --expt-relaxed-constexpr) 

set(HOST_OPT_FLAGS -Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qlanglvl=extended0x  -Xcompiler -qalias=noansi -Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-036)

set(HOST_RELDEB_FLAGS -Xcompiler -O2 -Xcompiler -g -Xcompiler -qstrict -Xcompiler -qsmp=omp:noopt -Xcompiler -qkeepparm -Xcompiler -qmaxmem=-1 -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 -Xcompiler -qsuppress=1500-030 -Xcompiler -qsuppress=1500-036 )

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJA_NVCC_FLAGS -O3; ${CUDA_COMMON_OPT_FLAGS}; ${HOST_OPT_FLAGS} CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJA_NVCC_FLAGS -g; -O2;  ${CUDA_COMMON_OPT_FLAGS};  ${HOST_RELDEB_FLAGS}  CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJA_NVCC_FLAGS -g; -G; -O0; ${CUDA_COMMON_DEBUG_FLAGS}; ${HOST_RELDEB_FLAGS} CACHE LIST "")
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")

