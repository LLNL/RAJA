//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Reduction policies used for reduction tests
//

#ifndef __RAJA_test_multi_reducepol_HPP__
#define __RAJA_test_multi_reducepol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential reduction policy types
using SequentialMultiReducePols = camp::list< RAJA::seq_multi_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPMultiReducePols =
  camp::list< RAJA::omp_multi_reduce,
              RAJA::omp_multi_reduce_ordered >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetMultiReducePols =
  camp::list< RAJA::omp_target_multi_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaMultiReducePols =
  camp::list< RAJA::cuda_multi_reduce_block_then_grid_atomic_host_init,
              RAJA::cuda_multi_reduce_global_atomic_host_init >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipMultiReducePols =
  camp::list< RAJA::hip_multi_reduce_block_then_grid_atomic_host_init,
              RAJA::hip_multi_reduce_global_atomic_host_init >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclMultiReducePols = camp::list< RAJA::sycl_multi_reduce >;
#endif

#endif  // __RAJA_test_multi_reducepol_HPP__
