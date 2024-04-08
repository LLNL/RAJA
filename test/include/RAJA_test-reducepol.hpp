//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Reduction policies used for reduction tests
//

#ifndef __RAJA_test_reducepol_HPP__
#define __RAJA_test_reducepol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential reduction policy types
using SequentialReducePols = camp::list< RAJA::seq_reduce >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPReducePols = 
#if 0 // is ordered reduction broken???
  camp::list< RAJA::omp_reduce,
              RAJA::omp_reduce_ordered >;
#else
  camp::list< RAJA::omp_reduce >;
#endif
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducePols =
  camp::list< RAJA::omp_target_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReducePols = camp::list< RAJA::cuda_reduce_with_fences,
                                   RAJA::cuda_reduce_avoid_fences,
                                   RAJA::cuda_reduce_atomic_with_fences,
                                   RAJA::cuda_reduce_atomic_avoid_fences,
                                   RAJA::cuda_reduce_atomic_host_with_fences,
                                   RAJA::cuda_reduce_atomic_host_avoid_fences >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipReducePols = camp::list< RAJA::hip_reduce_with_fences,
                                  RAJA::hip_reduce_avoid_fences,
                                  RAJA::hip_reduce_atomic_with_fences,
                                  RAJA::hip_reduce_atomic_avoid_fences,
                                  RAJA::hip_reduce_atomic_host_with_fences,
                                  RAJA::hip_reduce_atomic_host_avoid_fences >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclReducePols = camp::list< RAJA::sycl_reduce >;
#endif

#endif  // __RAJA_test_reducepol_HPP__
