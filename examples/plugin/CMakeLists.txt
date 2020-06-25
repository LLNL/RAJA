###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

raja_add_executable(
  NAME plugin-example
  SOURCES test-plugin.cpp counter-plugin.cpp)


raja_add_executable(
  NAME plugin-example-dynamic
  SOURCES test-plugin-dynamic.cpp)
  
add_library(timer_plugin SHARED timer-plugin.cpp)
target_link_libraries(timer_plugin PRIVATE RAJA)
