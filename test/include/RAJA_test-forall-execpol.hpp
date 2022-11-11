//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout forall tests
//

#ifndef __RAJA_test_forall_execpol_HPP__
#define __RAJA_test_forall_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

// Sequential execution policy types
using SequentialForallExecPols = camp::list< RAJA::seq_exec,
                                             RAJA::loop_exec,
                                             RAJA::simd_exec >;

//
// Sequential execution policy types for reduction and atomic tests.
//
// Note: RAJA::simd_exec does not work with these.
//
using SequentialForallReduceExecPols = camp::list< RAJA::seq_exec,
                                                   RAJA::loop_exec >;

using SequentialForallAtomicExecPols = camp::list< RAJA::seq_exec, 
                                                   RAJA::loop_exec >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPForallExecPols = 
  camp::list< RAJA::omp_parallel_for_exec
 
              , RAJA::omp_parallel_for_static_exec< >
              , RAJA::omp_parallel_for_static_exec<4>

#if defined(RAJA_TEST_EXHAUSTIVE)
              , RAJA::omp_parallel_for_dynamic_exec< >
              , RAJA::omp_parallel_for_dynamic_exec<4>

              , RAJA::omp_parallel_for_guided_exec< >
              , RAJA::omp_parallel_for_guided_exec<4>

              , RAJA::omp_parallel_for_runtime_exec

              , RAJA::omp_parallel_exec<RAJA::omp_for_exec>

              , RAJA::omp_parallel_exec<RAJA::omp_for_static_exec< >>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Static< >>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_static_exec<8>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Static<8>>>

              , RAJA::omp_parallel_exec<RAJA::omp_for_nowait_schedule_exec<RAJA::policy::omp::Static< >>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_nowait_static_exec<4>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_nowait_schedule_exec<RAJA::policy::omp::Static<4>>>

              , RAJA::omp_parallel_exec<RAJA::omp_for_dynamic_exec< >>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Dynamic< >>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_dynamic_exec<8>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Dynamic<8>>>

              , RAJA::omp_parallel_exec<RAJA::omp_for_guided_exec< >>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Guided< >>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_guided_exec<8>>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Guided<8>>>

              , RAJA::omp_parallel_exec<RAJA::omp_for_runtime_exec>
              , RAJA::omp_parallel_exec<RAJA::omp_for_schedule_exec<RAJA::policy::omp::Runtime>>
#endif       
             >;

using OpenMPForallReduceExecPols = OpenMPForallExecPols;

using OpenMPForallAtomicExecPols =
  camp::list< RAJA::omp_parallel_for_exec

#if defined(RAJA_TEST_EXHAUSTIVE)
              , RAJA::omp_parallel_for_static_exec< >
              , RAJA::omp_parallel_for_static_exec<4>
              , RAJA::omp_parallel_exec<RAJA::omp_for_nowait_static_exec< >>
              , RAJA::omp_parallel_exec<RAJA::omp_for_nowait_static_exec<4>>

              , RAJA::omp_parallel_for_dynamic_exec< >
              , RAJA::omp_parallel_for_dynamic_exec<2>

              , RAJA::omp_parallel_for_guided_exec< >
              , RAJA::omp_parallel_for_guided_exec<3>

              , RAJA::omp_parallel_for_runtime_exec
#endif
            >; 

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_TBB)
using TBBForallExecPols = camp::list< RAJA::tbb_for_exec,
                                      RAJA::tbb_for_static< >,
                                      RAJA::tbb_for_static< 2 >,
                                      RAJA::tbb_for_static< 4 >,
                                      RAJA::tbb_for_static< 8 >,
                                      RAJA::tbb_for_dynamic >;

using TBBForallReduceExecPols = TBBForallExecPols;

using TBBForallAtomicExecPols = TBBForallExecPols;

#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetForallExecPols =
  camp::list< RAJA::omp_target_parallel_for_exec<8>,
              RAJA::omp_target_parallel_for_exec_nt >;

using OpenMPTargetForallReduceExecPols = OpenMPTargetForallExecPols;

using OpenMPTargetForallAtomicExecPols = OpenMPTargetForallExecPols;

#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaForallExecPols = camp::list< RAJA::cuda_exec<128>,
                                       RAJA::cuda_exec<256>,
                                       RAJA::cuda_exec_explicit<256,2> >;

using CudaForallReduceExecPols = CudaForallExecPols;

using CudaForallAtomicExecPols = CudaForallExecPols;

#endif

#if defined(RAJA_ENABLE_HIP)
using HipForallExecPols = camp::list< RAJA::hip_exec<128>,
                                      RAJA::hip_exec<256>  >;

using HipForallReduceExecPols = HipForallExecPols;

using HipForallAtomicExecPols = HipForallExecPols;

#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclForallExecPols = camp::list< RAJA::sycl_exec<128, false>,
                                       RAJA::sycl_exec<256, false> >;

using SyclForallReduceExecPols = SyclForallExecPols;

using SyclForallAtomicExecPols = SyclForallExecPols;

#endif

#endif  // __RAJA_test_forall_execpol_HPP__
