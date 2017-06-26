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

SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

raja_add_test(
  NAME test-GPUreducemin
  SOURCES ReduceMin.cpp)

raja_add_test(
  NAME test-GPUreducemax
  SOURCES ReduceMax.cpp)

raja_add_test(
  NAME test-GPUreducesum
  SOURCES ReduceSum.cpp)

raja_add_test(
  NAME test-GPUforall
  SOURCES forall.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-GPUforallN
    SOURCES forallN.cpp)
endif()

raja_add_test(
  NAME test-GPUreduceminloc
  SOURCES ReduceMinLoc.cpp)

raja_add_test(
  NAME test-GPUreducemaxloc
  SOURCES ReduceMaxLoc.cpp)

raja_add_test(
  NAME test-GPUforallN-strided
  SOURCES forallN-strided.cpp
  DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

if (NOT RAJA_ENABLE_CLANG_CUDA)
  raja_add_test(
    NAME test-GPUscan
    SOURCES test-scan.cpp
    DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
endif ()
