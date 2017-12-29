###############################################################################
#
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

#raja_add_executable(
#  NAME waveShared2
#  SOURCES waveShared2.cpp)

#if(ENABLE_CUDA)
#raja_add_executable(
#  NAME cuda-shmem-ltimes
#  SOURCES cuda-shmem-ltimes.cpp)
#endif()

raja_add_executable(
  NAME cpu-shmem-ltimes
  SOURCES cpu-shmem-ltimes.cpp)
  