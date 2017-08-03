###############################################################################
#
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

include_directories(include)


set(test_macro__internal_dir ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "")

# use this macro to update the parent scope with the current list of sources used
# for testing. NOTE: this is only required when the last command in a CMakeLists.txt
# is add_directory(). add_gtest_sources() propagates the sources as expected
macro(propagate_sources)
  set (raja_test_SOURCES ${raja_test_SOURCES} PARENT_SCOPE)
endmacro()

# use this macro to add sources for testing. It will automatically add the
# relative path (if required). NOTE: this internally updates @raja_test_SOURCES
macro (add_gtest_sources)
  file (RELATIVE_PATH _relPath "${test_macro__internal_dir}" "${CMAKE_CURRENT_SOURCE_DIR}")
  foreach (_src ${ARGN})
    if (_relPath)
      list (APPEND raja_test_SOURCES "${_relPath}/${_src}")
    else()
      list (APPEND raja_test_SOURCES "${_src}")
    endif()
  endforeach()
  if (_relPath)
    propagate_sources()
  endif()
endmacro()

add_subdirectory(unit)

add_subdirectory(integration)

# create the test once the sources have been visited
raja_add_test(
  NAME raja_test_suite
  SOURCES ${raja_test_SOURCES})
