###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND PLUGIN_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND PLUGIN_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND PLUGIN_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND PLUGIN_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
  endif()
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
    NAME test-plugin-old-dynamic
    SOURCES test_plugin_old_dynamic.cpp)

  raja_add_plugin_library(NAME dynamic_plugin_old
                          SHARED TRUE
                          SOURCES plugin_old_for_test_dynamic.cpp)

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


if (RAJA_ENABLE_CUDA)
  # hack raja_add_test to not depend on cuda
  set(COPY_RAJA_ENABLE_CUDA ${RAJA_ENABLE_CUDA})
  set(RAJA_ENABLE_CUDA Off)
  raja_add_test(
    NAME test-inactive-gpu
    SOURCES test_inactive_gpu.cpp)
  set(RAJA_ENABLE_CUDA ${COPY_RAJA_ENABLE_CUDA})
endif ()

if (RAJA_ENABLE_HIP)
  # hack raja_add_test to not depend on hip
  set(COPY_RAJA_ENABLE_HIP ${RAJA_ENABLE_HIP})
  set(RAJA_ENABLE_HIP Off)
  raja_add_test(
    NAME test-inactive-gpu
    SOURCES test_inactive_gpu.cpp)
  set(RAJA_ENABLE_HIP ${COPY_RAJA_ENABLE_HIP})
endif ()

if (RAJA_ENABLE_SYCL)
  # hack raja_add_test to not depend on sycl
  set(COPY_RAJA_ENABLE_SYCL ${RAJA_ENABLE_SYCL})
  set(RAJA_ENABLE_SYCL Off)
  raja_add_test(
    NAME test-inactive-gpu
    SOURCES test_inactive_gpu.cpp)
  set(RAJA_ENABLE_SYCL ${COPY_RAJA_ENABLE_SYCL})
endif ()

if (RAJA_ENABLE_TARGET_OPENMP)
  # hack raja_add_test to not depend on target openmp
  set(COPY_RAJA_ENABLE_TARGET_OPENMP ${RAJA_ENABLE_TARGET_OPENMP})
  set(RAJA_ENABLE_TARGET_OPENMP Off)
  raja_add_test(
    NAME test-inactive-gpu
    SOURCES test_inactive_gpu.cpp)
  set(RAJA_ENABLE_TARGET_OPENMP ${COPY_RAJA_ENABLE_TARGET_OPENMP})
endif ()
