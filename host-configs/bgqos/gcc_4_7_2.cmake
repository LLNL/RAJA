##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For release details and restrictions, please see RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

set(CMAKE_CXX_COMPILER "/usr/local/tools/compilers/ibm/mpicxx-4.7.2b" CACHE PATH "")

set(TEST_DRIVER srun CACHE STRING "use slurm to launch on BGQ")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Ofast -mcpu=a2 -mtune=a2 -finline-functions -finline-limit=20000 -std=c++11" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -std=c++11" CACHE STRING "")

set(ENABLE_OPENMP On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
