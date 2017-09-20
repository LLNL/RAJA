###############################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#    
# Produced at the Lawrence Livermore National Laboratory
#    
# LLNL-CODE-689114
# 
# All rights reserved.
#  
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")
#set(RAJA_FP "RAJA_USE_FLOAT")
option(RAJA_USE_DOUBLE On)
option(RAJA_USE_FLOAT Off)
option(RAJA_USE_COMPLEX Off)

## Pointer options
if (RAJA_ENABLE_CUDA)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance options
option(RAJA_ENABLE_FT "Enable fault-tolerance features" OFF)
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
  if(RAJA_ENABLE_CUDA)
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
