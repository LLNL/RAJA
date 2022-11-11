###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_ENABLE_OPENMP)
  if(OPENMP_FOUND)
    list(APPEND RAJA_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(RAJA_ENABLE_OPENMP Off)
  endif()
endif()

if (RAJA_ENABLE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    blt_register_library(
      NAME tbb
      INCLUDES ${TBB_INCLUDE_DIRS}
      LIBRARIES ${TBB_LIBRARIES})
    message(STATUS "TBB Enabled")
  else()
    message(WARNING "TBB NOT FOUND")
    set(RAJA_ENABLE_TBB Off CACHE BOOL "" FORCE)
  endif()
endif ()

if (RAJA_ENABLE_CUDA)
  if (RAJA_ENABLE_EXTERNAL_CUB STREQUAL "VersionDependent")
    if (CUDA_VERSION_STRING VERSION_GREATER_EQUAL "11.0")
      set(RAJA_ENABLE_EXTERNAL_CUB ON)
      message(STATUS "Setting RAJA_ENABLE_EXTERNAL_CUB ON with CUDA_VERSION ${CUDA_VERSION_STRING} >= 11.")
    else()
      set(RAJA_ENABLE_EXTERNAL_CUB OFF)
      message(STATUS "Setting RAJA_ENABLE_EXTERNAL_CUB OFF with CUDA_VERSION ${CUDA_VERSION_STRING} < 11.")
    endif()
  endif()

  if (RAJA_ENABLE_EXTERNAL_CUB)
    find_package(CUB)
    if (CUB_FOUND)
      blt_import_library(
        NAME cub
        INCLUDES ${CUB_INCLUDE_DIRS}
        TREAT_INCLUDES_AS_SYSTEM ON
        EXPORTABLE ON)
    else()
      message(FATAL_ERROR "External CUB not found, CUB_DIR=${CUB_DIR}.")
    endif()
  else()
    message(STATUS "Using RAJA CUB submodule.")
  endif()
endif ()

if (RAJA_ENABLE_CUDA AND RAJA_ENABLE_NV_TOOLS_EXT)
  find_package(NvToolsExt)
  if (NVTOOLSEXT_FOUND)
    blt_import_library( NAME       nvtoolsext
                        TREAT_INCLUDES_AS_SYSTEM ON
                        INCLUDES   ${NVTOOLSEXT_INCLUDE_DIRS}
                        LIBRARIES  ${NVTOOLSEXT_LIBRARY}
                        EXPORTABLE ON
                      )
  else()
    message(FATAL_ERROR "NvToolsExt not found, NVTOOLSEXT_DIR=${NVTOOLSEXT_DIR}.")
  endif()
endif ()

if (RAJA_ENABLE_HIP)
  if (RAJA_ENABLE_EXTERNAL_ROCPRIM STREQUAL "VersionDependent")
    if (hip_VERSION VERSION_GREATER_EQUAL "4.0")
      set(RAJA_ENABLE_EXTERNAL_ROCPRIM ON)
      message(STATUS "Setting RAJA_ENABLE_EXTERNAL_ROCPRIM ON with hip_VERSION ${hip_VERSION} >= 4.")
    else()
      set(RAJA_ENABLE_EXTERNAL_ROCPRIM OFF)
      message(STATUS "Setting RAJA_ENABLE_EXTERNAL_ROCPRIM OFF with hip_VERSION ${hip_VERSION} < 4.")
    endif()
  endif()

  if (RAJA_ENABLE_EXTERNAL_ROCPRIM)
    include(cmake/thirdparty/FindRocPRIM.cmake)
    if (ROCPRIM_FOUND)
      blt_import_library(
        NAME rocPRIM
        INCLUDES ${ROCPRIM_INCLUDE_DIRS}
        TREAT_INCLUDES_AS_SYSTEM ON
        EXPORTABLE ON)
    else()
      message(FATAL_ERROR "External rocPRIM not found, ROCPRIM_DIR=${ROCPRIM_DIR}.")
    endif()
  else()
    message(STATUS "Using RAJA rocPRIM submodule.")
  endif()
endif ()

if (RAJA_ENABLE_HIP AND RAJA_ENABLE_ROCTX)
  include(FindRoctracer)
  blt_import_library(NAME roctx
                     INCLUDES ${ROCTX_INCLUDE_DIRS}
                     LIBRARIES ${ROCTX_LIBRARIES}
                     EXPORTABLE ON
                     TREAT_INCLUDES_AS_SYSTEM ON)
endif ()

set(TPL_DEPS)
blt_list_append(TO TPL_DEPS ELEMENTS nvtoolsext IF RAJA_ENABLE_NV_TOOLS_EXT)
blt_list_append(TO TPL_DEPS ELEMENTS cub IF RAJA_ENABLE_EXTERNAL_CUB)
blt_list_append(TO TPL_DEPS ELEMENTS rocPRIM IF RAJA_ENABLE_EXTERNAL_ROCPRIM)
blt_list_append(TO TPL_DEPS ELEMENTS roctx IF RAJA_ENABLE_ROCTX)

set(RAJA_NEEDS_BLT_TPLS False)
if (RAJA_ENABLE_CUDA OR RAJA_ENABLE_HIP OR RAJA_ENABLE_OPENMP OR RAJA_ENABLE_MPI)
  set(RAJA_NEEDS_BLT_TPLS True)
endif ()

if (RAJA_NEEDS_BLT_TPLS)
  if (NOT BLT_EXPORTED)
    set(BLT_EXPORTED On CACHE BOOL "" FORCE)
  blt_import_library(NAME          blt_stub EXPORTABLE On)
  set_target_properties(blt_stub PROPERTIES EXPORT_NAME blt::blt_stub)
            install(TARGETS blt_stub
                    EXPORT               bltTargets)
    blt_export_tpl_targets(EXPORT bltTargets NAMESPACE blt)
    install(EXPORT bltTargets
      DESTINATION  lib/cmake/raja)
  endif()
endif ()

foreach(dep ${TPL_DEPS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        install(TARGETS              ${dep}
                EXPORT               RAJATargets
                DESTINATION          lib/cmake/raja)
        # Namespace target to avoid conflicts
        set_target_properties(${dep} PROPERTIES EXPORT_NAME RAJA::${dep})
    endif()
endforeach()
