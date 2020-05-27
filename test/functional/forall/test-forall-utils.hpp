//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_UTILS_HPP__
#define __TEST_FORALL_UTILS_HPP__

#include "RAJA/RAJA.hpp"

#include "camp/resource.hpp"
#include "camp/list.hpp"

#include "gtest/gtest.h"

//
// Unroll types for gtest testing::Types
//
template <class T>
struct Test;

template <class... T>
struct Test<camp::list<T...>> {
  using Types = ::testing::Types<T...>;
};

//
// Strongly typed indexes
//
RAJA_INDEX_VALUE(StrongIndexType, "StrongIndexType");
RAJA_INDEX_VALUE_T(StrongInt, int, "StrongIntType");
RAJA_INDEX_VALUE_T(StrongULL, unsigned long long , "StrongULLType");

//
// Index types list
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
// Index types w/ Strong types list
//
using StrongIdxTypeList = camp::list<RAJA::Index_type,
                                     int,
                                     StrongIndexType,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                     StrongInt,
                                     unsigned int,
                                     short,
                                     unsigned short,
                                     long int,
                                     unsigned long,
                                     long long,
#endif
                                     StrongULL,
                                     unsigned long long>;


//
// Memory resource types for beck-end execution
//
using HostResourceList = camp::list<camp::resources::Host>;

#if defined(RAJA_ENABLE_CUDA)
using CudaResourceList = camp::list<camp::resources::Cuda>;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetResourceList = camp::list<camp::resources::Omp>;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipResourceList = camp::list<camp::resources::Hip>;
#endif


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

  *work_array = work_res.allocate<T>(RAJA::stripIndexType(N));

  *check_array = host_res.allocate<T>(RAJA::stripIndexType(N));
  *test_array = host_res.allocate<T>(RAJA::stripIndexType(N));
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
