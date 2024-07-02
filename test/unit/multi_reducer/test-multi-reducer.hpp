//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_MULTI_REDUCER_UTILS_HPP__
#define __TEST_MULTI_REDUCER_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"

#include "RAJA_unit-test-forone.hpp"
#include "RAJA_test-multi-reduce-abstractor.hpp"

//
// Data types
//
using DataTypeList = camp::list< int,
                                 float,
                                 double >;

using SequentialMultiReducerPolicyList = camp::list< RAJA::seq_multi_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPMultiReducerPolicyList = camp::list< RAJA::omp_multi_reduce,
                                                 RAJA::omp_multi_reduce_ordered >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetMultiReducerPolicyList = camp::list< RAJA::omp_target_multi_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaMultiReducerPolicyList =
  camp::list< RAJA::cuda_multi_reduce_block_then_grid_atomic_host_init,
              RAJA::cuda_multi_reduce_global_atomic_host_init >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipMultiReducerPolicyList =
  camp::list< RAJA::hip_multi_reduce_block_then_grid_atomic_host_init,
              RAJA::hip_multi_reduce_global_atomic_host_init >;
#endif

#endif  // __TEST_MULTI_REDUCER_UTILS_HPP__
