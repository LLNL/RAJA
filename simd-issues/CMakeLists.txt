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

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -c -S -qopt-report=5 -qopt-report-phase=loop,vec -qopt-report-file=stderr")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")

raja_add_executable(
  NAME example1
  SOURCES example1.cpp)

raja_add_executable(
  NAME example2
  SOURCES example2.cpp)
