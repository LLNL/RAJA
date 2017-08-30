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

macro(raja_add_executable)
  set(options )
  set(singleValueArgs NAME TEST)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON RAJA)

  if (ENABLE_CHAI)
    list (APPEND arg_DEPENDS_ON chai)
  endif ()

  if (ENABLE_OPENMP)
    list (APPEND arg_DEPENDS_ON openmp)
  endif ()

  if (ENABLE_CUDA)
    list (APPEND arg_DEPENDS_ON cuda)
  endif ()

  message(STATUS "${arg_NAME} building with depends: ${arg_DEPENDS_ON}")

  if (${arg_TEST})
    set (_output_dir test)
  else ()
    set (_output_dir bin)
  endif()

  blt_add_executable(
    NAME ${arg_NAME}
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    OUTPUT_DIR ${_output_dir}
    )
endmacro(raja_add_executable)

macro(raja_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    TEST On)

  blt_add_test(
    NAME ${arg_NAME}
    #COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
    COMMAND ${TEST_DRIVER} ${arg_NAME})
endmacro(raja_add_test)
