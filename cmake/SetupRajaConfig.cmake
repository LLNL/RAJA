###############################################################################
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")
#set(RAJA_FP "RAJA_USE_FLOAT")
option(RAJA_USE_DOUBLE On)
option(RAJA_USE_FLOAT Off)
option(RAJA_USE_COMPLEX Off)

## Pointer options
if (ENABLE_CUDA)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance options
option(ENABLE_FT "Enable fault-tolerance features" OFF)
option(RAJA_REPORT_FT "Report on use of fault-tolerant features" OFF)

## Timer options
set(RAJA_TIMER "chrono" CACHE STRING
    "Select a timer backend")
set_property(CACHE RAJA_TIMER PROPERTY STRINGS "chrono" "gettime" "clock" )

if (RAJA_TIMER STREQUAL "chrono")
    set(RAJA_USE_CHRONO  ON  CACHE BOOL "Use the default std::chrono timer" )
else ()
    set(RAJA_USE_CHRONO  OFF  CACHE BOOL "Use the default std::chrono timer" )
endif ()
if (RAJA_TIMER STREQUAL "gettime")
    set(RAJA_USE_GETTIME ON CACHE BOOL "Use clock_gettime from time.h for timer"      )
else ()
    set(RAJA_USE_GETTIME OFF CACHE BOOL "Use clock_gettime from time.h for timer"       )
endif ()
if (RAJA_TIMER STREQUAL "clock")
    set(RAJA_USE_CLOCK   ON CACHE BOOL "Use clock from time.h for timer"     )
else ()
    set(RAJA_USE_CLOCK   OFF CACHE BOOL "Use clock from time.h for timer"    )
endif ()

include(CheckFunctionExists)
check_function_exists(posix_memalign RAJA_HAVE_POSIX_MEMALIGN)
check_function_exists(aligned_alloc RAJA_HAVE_ALIGNED_ALLOC)
check_function_exists(_mm_malloc RAJA_HAVE_MM_MALLOC)

# Set up RAJA_ENABLE prefixed options
set(RAJA_ENABLE_OPENMP ${ENABLE_OPENMP})
set(RAJA_ENABLE_TARGET_OPENMP ${ENABLE_TARGET_OPENMP})
set(RAJA_ENABLE_TBB ${ENABLE_TBB})
set(RAJA_ENABLE_CUDA ${ENABLE_CUDA})
set(RAJA_ENABLE_CLANG_CUDA ${ENABLE_CLANG_CUDA})
set(RAJA_ENABLE_CHAI ${ENABLE_CHAI})
set(RAJA_ENABLE_CUB ${ENABLE_CUB})

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp)

# Configure CMake config
configure_file(${PROJECT_SOURCE_DIR}/share/raja/cmake/RAJA-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/raja/cmake/raja-config.cmake)

install(FILES ${PROJECT_BINARY_DIR}/share/raja/cmake/raja-config.cmake
  DESTINATION share/raja/cmake/)

# Setup pkg-config
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # convert lists of link libraries into -lstdc++ -lm etc..
  foreach(LIB ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${PLATFORM_LIBS} ${CUDA_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} -l${LIB}")
  endforeach()
  foreach(INCDIR ${INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    set(PC_C_FLAGS "${PC_C_FLAGS} -I${INCDIR}")
  endforeach()
  if(ENABLE_CUDA)
    foreach(FLAG ${RAJA_NVCC_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  else()
    foreach(FLAG ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  endif()
  # Produce a pkg-config file for linking against the shared lib
  configure_file("share/raja/pkg-config/RAJA.pc.in" "RAJA.pc" @ONLY)
  install(FILES       "${CMAKE_CURRENT_BINARY_DIR}/RAJA.pc"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")
endif()
