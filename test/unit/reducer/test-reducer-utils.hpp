//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_REDUCER_UTILS_HPP__
#define __TEST_REDUCER_UTILS_HPP__

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
// Data types
//
using DataTypeList = camp::list<
                                 int,
                                 float,
                                 double
                               >;

using HostReducerPolicyList =
    camp::list<
                RAJA::seq_reduce

#if defined(RAJA_ENABLE_TBB)
                ,
                RAJA::tbb_reduce
#endif

#if defined(RAJA_ENABLE_OPENMP)
                ,
                RAJA::omp_reduce,
                RAJA::omp_reduce_ordered
#endif
              >;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducerPolicyList = camp::list< RAJA::omp_target_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReducerPolicyList = camp::list< RAJA::cuda_reduce >;
#endif


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

#endif  // __TEST_REDUCER_UTILS_HPP__
