//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_UTILS_HPP__
#define __TEST_UTILS_HPP__

#include "camp/resource.hpp"
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

#endif  // __TEST_UTILS_HPP__
