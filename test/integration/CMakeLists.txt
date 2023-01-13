###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND PLUGIN_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND PLUGIN_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND PLUGIN_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND PLUGIN_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND PLUGIN_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  #  list(APPEND PLUGIN_BACKENDS OpenMPTarget)
endif()

add_subdirectory(plugin)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  if(NOT WIN32)
  raja_add_test(
    NAME test-plugin-dynamic
    SOURCES test_plugin_dynamic.cpp)

  raja_add_plugin_library(NAME dynamic_plugin
                          SHARED TRUE
                          SOURCES plugin_for_test_dynamic.cpp)

  raja_add_test(
    NAME test-plugin-kokkos
    SOURCES test_plugin_kokkos.cpp)

  raja_add_plugin_library(NAME kokkos_plugin
                          SHARED TRUE
                          SOURCES plugin_for_test_kokkos.cpp)

  set_tests_properties(test-plugin-kokkos.exe PROPERTIES
                      ENVIRONMENT "KOKKOS_PLUGINS=${CMAKE_BINARY_DIR}/lib/libkokkos_plugin.so")
  endif()
endif ()
