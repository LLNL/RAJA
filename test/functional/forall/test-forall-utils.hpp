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

#if defined(RAJA_ENABLE_SYCL)
using SyclResourceList = camp::list<camp::resources::Sycl>;
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
