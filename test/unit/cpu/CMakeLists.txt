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
# For additional details, please also read raja/README-license.txt.
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

if (NOT RAJA_ENABLE_NESTED)
  add_definitions(-DRAJA_ENABLE_NESTED)
endif()

if (RAJA_ENABLE_CUDA)
  cuda_add_library(bis buildIndexSet.cxx)
  cuda_add_executable(CPUtraversal-test.exe
    main-traversal.cxx)

  cuda_add_executable(CPUreduce-test.exe
    main-reduce.cxx)

  cuda_add_executable(CPUnested-test.exe
    main-nested.cxx)

  if(RAJA_ENABLE_OPENMP)
      cuda_add_executable(CPUnested_reduce-test.exe
        main-nested-reduce.cxx)
  endif()

  cuda_add_executable(indexset-tests
    test-indexsets.cxx)

  cuda_add_executable(traversal-tests
    test-traversals.cxx)

  cuda_add_executable(scan-tests
    test-scan.cxx)

else()
  add_library(bis buildIndexSet.cxx)
  add_executable(CPUtraversal-test.exe
    main-traversal.cxx)

  add_executable(CPUreduce-test.exe
    main-reduce.cxx)

  add_executable(CPUnested-test.exe
    main-nested.cxx)

  if(RAJA_ENABLE_OPENMP)
    add_executable(CPUnested_reduce-test.exe
      main-nested-reduce.cxx)
  endif()

  add_executable(indexset-tests
    test-indexsets.cxx)

  add_executable(traversal-tests
    test-traversals.cxx)

  add_executable(scan-tests
    test-scan.cxx)

endif()

add_test(
  NAME CPUtraversal
  COMMAND ${TEST_DRIVER} $<TARGET_FILE:CPUtraversal-test.exe>)

add_test(
  NAME CPUreduce
  COMMAND ${TEST_DRIVER} $<TARGET_FILE:CPUreduce-test.exe>)

add_test(
  NAME CPUnested
  COMMAND ${TEST_DRIVER} $<TARGET_FILE:CPUnested-test.exe>)

if(RAJA_ENABLE_OPENMP)
    add_test(
      NAME CPUnested-reduce
      COMMAND ${TEST_DRIVER} $<TARGET_FILE:CPUnested_reduce-test.exe>)
endif()

add_test(
    NAME indexset-test
    COMMAND ${TEST_DRIVER} $<TARGET_FILE:indexset-tests>)

target_link_libraries(indexset-tests RAJA bis gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

add_test(
    NAME traversal-test
    COMMAND ${TEST_DRIVER} $<TARGET_FILE:traversal-tests>)

target_link_libraries(traversal-tests RAJA bis gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

add_test(
    NAME scan-test
    COMMAND ${TEST_DRIVER} $<TARGET_FILE:scan-tests>)

target_link_libraries(scan-tests RAJA gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(CPUtraversal-test.exe RAJA bis)
target_link_libraries(CPUreduce-test.exe RAJA bis)
target_link_libraries(CPUnested-test.exe RAJA)
if(RAJA_ENABLE_OPENMP)
  target_link_libraries(CPUnested_reduce-test.exe RAJA)
endif()
