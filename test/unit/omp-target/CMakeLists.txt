###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

if(ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-omp-target-nested-reduce
    SOURCES test-nested-reduce.cpp)
  raja_add_test(
    NAME test-omp-target-reductions
    SOURCES test-reductions.cpp)
endif(ENABLE_TARGET_OPENMP)
