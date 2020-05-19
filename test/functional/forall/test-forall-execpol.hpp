//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_EXECPOL_HPP__
#define __TEST_FORALL_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"

// Sequential execution policy types
using SequentialForallExecPols = camp::list< RAJA::seq_exec,
                                             RAJA::loop_exec,
                                             RAJA::simd_exec >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPForallExecPols = 
  camp::list< // This policy works for the tests, but commenting it out
              // since its usage is questionable
              // RAJA::omp_parallel_exec<RAJA::seq_exec>,
              RAJA::omp_for_nowait_exec,
              RAJA::omp_for_exec,
              RAJA::omp_parallel_for_exec >;
#endif

#if defined(RAJA_ENABLE_TBB)
using TBBForallExecPols = camp::list< RAJA::tbb_for_exec,
                                      RAJA::tbb_for_static< >,
                                      RAJA::tbb_for_static< 2 >,
                                      RAJA::tbb_for_static< 4 >,
                                      RAJA::tbb_for_static< 8 >,
                                      RAJA::tbb_for_dynamic >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetForallExecPols =
  camp::list< RAJA::omp_target_parallel_for_exec<8>,
              RAJA::omp_target_parallel_for_exec_nt >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaForallExecPols = camp::list< RAJA::cuda_exec<128>,
                                       RAJA::cuda_exec<256> >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipForallExecPols = camp::list< RAJA::hip_exec<128>,
                                      RAJA::hip_exec<256>  >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclForallExecPols = camp::list< RAJA::sycl_exec<128>,
                                       RAJA::sycl_exec<256>  >;
#endif

#endif  // __TEST_FORALL_EXECPOL_HPP__
