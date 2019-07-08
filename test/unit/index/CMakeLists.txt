###############################################################################
#
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
raja_add_test(
  NAME test-rangesegment
  SOURCES test-rangesegment.cpp)

raja_add_test(
  NAME test-rangestridesegment
  SOURCES test-rangestridesegment.cpp)

raja_add_test(
  NAME test-listsegment
  SOURCES test-listsegment.cpp)
