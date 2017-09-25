###############################################################################
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

if (RAJA_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND RAJA_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(RAJA_ENABLE_OPENMP Off)
  endif()
endif()

if (RAJA_ENABLE_CLANG_CUDA)
  set(RAJA_ENABLE_CUDA On)
endif ()

if (RAJA_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  set (CUDA_PROPAGATE_HOST_FLAGS OFF)
  include_directories(${CUDA_INCLUDE_DIRS})

  if (RAJA_ENABLE_CUB)

    find_package(CUB)

    if (CUB_FOUND)
      include_directories(${CUB_INCLUDE_DIRS})
    else()
      message(WARNING "Using deprecated Thrust backend for CUDA scans.\n
  Please set CUB_DIR for better scan performance.")
      set(RAJA_ENABLE_CUB False)
    endif()
  endif()
endif()


if (RAJA_ENABLE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    message(STATUS "TBB Enabled")
  else()
    message(WARNING "TBB NOT FOUND")
    set(RAJA_ENABLE_TBB Off)
  endif()
endif ()

if (RAJA_ENABLE_TESTS)

#
# This conditional prevents build problems resulting from BLT and
# RAJA each having their own copy of googletest.
#
if (RAJA_BUILD_WITH_BLT)
else()

  include(ExternalProject)
  # Set default ExternalProject root directory
  SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/tpl)

  ExternalProject_Add(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.7.0
      CMAKE_ARGS                
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DCMAKE_CXX_COMPILER_ARG1=${CMAKE_CXX_COMPILER_ARG1}
      INSTALL_COMMAND ""
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON)

  ExternalProject_Get_Property(googletest source_dir)
  include_directories(${source_dir}/include)

  ExternalProject_Get_Property(googletest binary_dir)
  add_library(gtest      UNKNOWN IMPORTED)
  add_library(gtest_main UNKNOWN IMPORTED)

  if ( UNIX )
    set_target_properties(gtest PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/libgtest.a
    )
    set_target_properties(gtest_main PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/libgtest_main.a
    )
  elseif( WIN32 )
    set_target_properties(gtest PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_BUILD_TYPE}/gtest.lib
    )
    set_target_properties(gtest_main PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_BUILD_TYPE}/gtest_main.lib
    )
  endif ()
  add_dependencies(gtest      googletest)
  add_dependencies(gtest_main googletest)

  # GoogleTest requires threading
  find_package(Threads)

  enable_testing()
endif ()

endif ()

if (RAJA_ENABLE_DOCUMENTATION)
  find_package(Sphinx)
  find_package(Doxygen)
endif ()

if (RAJA_ENABLE_CHAI)
  message(STATUS "CHAI enabled")

  find_package(chai)

  include_directories(${CHAI_INCLUDE_DIRS})
endif()
