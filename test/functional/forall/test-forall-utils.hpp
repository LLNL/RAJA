//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_UTILS_HPP__
#define __TEST_FORALL_UTILS_HPP__

#include "camp/resource.hpp"
#include "camp/list.hpp"

#include "gtest/gtest.h"
#include "test-utils.hpp"


//
// Index types for segments
//
using IdxTypeList = camp::list<RAJA::Index_type,
                               int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                               unsigned int,
                               short,
                               unsigned short,
                               long int,
                               unsigned long,
                               long long,
#endif
                               unsigned long long>;


//
// Memory allocation/deallocation  methods for test execution
//

template<typename T>
void allocateForallTestData(T N,
                            camp::resources::Resource& work_res,
                            T** work_array,
                            T** check_array,
                            T** test_array)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  *work_array = work_res.allocate<T>(N);

  *check_array = host_res.allocate<T>(N);
  *test_array = host_res.allocate<T>(N);
}

template<typename T>
void deallocateForallTestData(camp::resources::Resource& work_res,
                              T* work_array,
                              T* check_array,
                              T* test_array)
{
  camp::resources::Resource host_res{camp::resources::Host()};

  work_res.deallocate(work_array);

  host_res.deallocate(check_array);
  host_res.deallocate(test_array);
}

#endif  // __TEST_FORALL_UTILS_HPP__
