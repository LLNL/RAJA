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

if (ENABLE_CUDA)
  raja_add_test(
    NAME test-chai-nested
    SOURCES chai-nested.cpp)
endif(ENABLE_CUDA)

raja_add_test(
  NAME test-chai-policy
  SOURCES chai-policy-tests.cpp)

raja_add_test(
  NAME test-chai
  SOURCES chai-tests.cpp)
