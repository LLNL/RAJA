//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Reduction policies used for reduction tests
//

#ifndef RAJA_test_reducepol_HPP
#define RAJA_test_reducepol_HPP

#include "RAJA/RAJA.hpp"

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

#if defined(RAJA_ENABLE_TBB)
using TBBReducePols = camp::list< RAJA::tbb_reduce >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetReducePols =
  camp::list< RAJA::omp_target_reduce >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaReducePols = camp::list< RAJA::cuda_reduce >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipReducePols = camp::list< RAJA::hip_reduce >;
#endif

#endif  // RAJA_test_reducepol_HPP
