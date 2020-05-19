//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_INDEXSET_EXECPOL_HPP__
#define __TEST_FORALL_INDEXSET_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"

// Sequential execution policy types
using SequentialForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec> >;

#if defined(RAJA_ENABLE_OPENMP)
using OpenMPForallIndexSetExecPols =  
  camp::list< RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::omp_parallel_segit, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_nowait_exec> >;
#endif

#if defined(RAJA_ENABLE_TBB)
using TBBForallIndexSetExecPols = 
  camp::list< RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_exec, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::seq_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::loop_exec>,
              RAJA::ExecPolicy<RAJA::tbb_for_dynamic, RAJA::simd_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_exec>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 2 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 4 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_static< 8 >>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::tbb_for_dynamic> >;
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using OpenMPTargetForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit,
                               RAJA::omp_target_parallel_for_exec<8>>,
              RAJA::ExecPolicy<RAJA::seq_segit, 
                               RAJA::omp_target_parallel_for_exec_nt> >;
#endif

#if defined(RAJA_ENABLE_CUDA)
using CudaForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<256>> >;
#endif

#if defined(RAJA_ENABLE_HIP)
using HipForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::hip_exec<256>> >;
#endif

#if defined(RAJA_ENABLE_SYCL)
using SyclForallIndexSetExecPols =
  camp::list< RAJA::ExecPolicy<RAJA::seq_segit, RAJA::sycl_exec<128>>,
              RAJA::ExecPolicy<RAJA::seq_segit, RAJA::sycl_exec<256>> >;
#endif

#endif  // __TEST_FORALL_INDEXSET_EXECPOL_HPP__
