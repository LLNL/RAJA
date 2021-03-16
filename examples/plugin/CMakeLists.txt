###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

raja_add_executable(
  NAME plugin-example
  SOURCES test-plugin.cpp counter-plugin.cpp)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  raja_add_executable(
    NAME plugin-example-dynamic
    SOURCES test-plugin-dynamic.cpp)
    
  raja_add_plugin_library(NAME timer_plugin
                          SHARED TRUE
                          SOURCES timer-plugin.cpp)
endif ()
