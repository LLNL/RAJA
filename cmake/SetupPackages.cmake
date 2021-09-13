###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
    set(RAJA_ENABLE_TBB Off)
  endif()
endif ()

if (RAJA_ENABLE_CUDA OR RAJA_ENABLE_EXTERNAL_CUB)
  find_package(CUB)
  if (CUB_FOUND)
    set(RAJA_ENABLE_EXTERNAL_CUB On)
    blt_import_library(
      NAME cub
      INCLUDES ${CUB_INCLUDE_DIRS}
      TREAT_INCLUDES_AS_SYSTEM ON
      EXPORTABLE ON)
  elseif(RAJA_ENABLE_EXTERNAL_CUB)
    message(FATAL_ERROR "External CUB not found, CUB_DIR=${CUB_DIR}.")
  else()
    message(STATUS "Using RAJA CUB submodule.")
  endif()
endif ()

if (RAJA_ENABLE_CUDA AND ENABLE_NV_TOOLS_EXT)
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

if (ENABLE_HIP OR RAJA_ENABLE_EXTERNAL_ROCPRIM)
  find_package(RocPRIM)
  if (ROCPRIM_FOUND)
    set(RAJA_ENABLE_EXTERNAL_ROCPRIM On)
    blt_import_library(
      NAME rocPRIM
      INCLUDES ${ROCPRIM_INCLUDE_DIRS}
      TREAT_INCLUDES_AS_SYSTEM ON
      EXPORTABLE ON)
  elseif (RAJA_ENABLE_EXTERNAL_ROCPRIM)
      message(FATAL_ERROR "External rocPRIM not found, ROCPRIM_DIR=${ROCPRIM_DIR}.")
  else()
    message(STATUS "Using RAJA rocPRIM submodule.")
  endif()
endif ()

set(TPL_DEPS)
blt_list_append(TO TPL_DEPS ELEMENTS cuda cuda_runtime IF RAJA_ENABLE_CUDA)
blt_list_append(TO TPL_DEPS ELEMENTS nvtoolsext IF ENABLE_NV_TOOLS_EXT)
blt_list_append(TO TPL_DEPS ELEMENTS cub IF RAJA_ENABLE_EXTERNAL_CUB)
blt_list_append(TO TPL_DEPS ELEMENTS hip hip_runtime IF ENABLE_HIP)
blt_list_append(TO TPL_DEPS ELEMENTS rocPRIM IF RAJA_ENABLE_EXTERNAL_ROCPRIM)
blt_list_append(TO TPL_DEPS ELEMENTS openmp IF RAJA_ENABLE_OPENMP)
blt_list_append(TO TPL_DEPS ELEMENTS mpi IF ENABLE_MPI)

foreach(dep ${TPL_DEPS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        install(TARGETS              ${dep}
                EXPORT               RAJA
                DESTINATION          lib)
        # Namespace target to avoid conflicts
        set_target_properties(${dep} PROPERTIES EXPORT_NAME RAJA::${dep})
    endif()
endforeach()
