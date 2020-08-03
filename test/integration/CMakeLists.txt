###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
  list(APPEND PLUGIN_BACKENDS OpenMPTarget)
endif()

add_subdirectory(plugin)

raja_add_test(
  NAME test-plugin-dynamic
  SOURCES test_plugin_dynamic.cpp)

add_library(dynamic_plugin SHARED plugin_for_test_dynamic.cpp)
target_link_libraries(dynamic_plugin PRIVATE RAJA gtest)

raja_add_test(
  NAME test-plugin-kokkos
  SOURCES test_plugin_kokkos.cpp)

add_library(kokkos_plugin SHARED plugin_for_test_kokkos.cpp)
target_link_libraries(kokkos_plugin PRIVATE RAJA gtest)

set_tests_properties(test-plugin-kokkos.exe PROPERTIES
                     ENVIRONMENT "KOKKOS_PLUGINS=${CMAKE_BINARY_DIR}/lib/libkokkos_plugin.so")
