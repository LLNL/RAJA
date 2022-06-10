//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout teams tests
//

#ifndef __RAJA_TEST_TEAMS_RUNTIME_EXECPOL_HPP__
#define __RAJA_TEST_TEAMS_RUNTIME_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//Launch policies
#if defined(RAJA_ENABLE_CUDA)
using seq_cuda_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t,RAJA::expt::cuda_launch_t<true>>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::cuda_block_x_direct>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_thread_x_loop>>;

using seq_cuda_explicit_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t,RAJA::policy::cuda::expt::cuda_launch_explicit_t<true, 0, 0>>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::cuda_block_x_direct>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_thread_x_loop>>;

using Sequential_launch_policies = camp::list<
        seq_cuda_policies,
        seq_cuda_explicit_policies
         >;

#elif defined(RAJA_ENABLE_HIP)
using seq_hip_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t,RAJA::expt::hip_launch_t<true>>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::hip_block_x_direct>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::hip_thread_x_loop>>;

using Sequential_launch_policies = camp::list<
         seq_hip_policies
         >;
#else
using Sequential_launch_policies = camp::list<
        camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec>>>;
#endif // Sequential


#if defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_CUDA)

using omp_cuda_policies = camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t,RAJA::expt::cuda_launch_t<false>>,
         RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec, RAJA::cuda_block_x_direct>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_thread_x_loop>
  >;

using omp_cuda_explicit_policies = camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t,RAJA::policy::cuda::expt::cuda_launch_explicit_t<false, 0, 0>>,
         RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec, RAJA::cuda_block_x_direct>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::cuda_thread_x_loop>
  >;

using OpenMP_launch_policies = camp::list<
         omp_cuda_policies,
         omp_cuda_explicit_policies
         >;

#elif defined(RAJA_ENABLE_HIP)

using omp_hip_policies = camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t,RAJA::expt::hip_launch_t<false>>,
         RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec, RAJA::hip_block_x_direct>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec,RAJA::hip_thread_x_loop>
  >;

using OpenMP_launch_policies = camp::list<
         omp_hip_policies
         >;
#else
using OpenMP_launch_policies = camp::list<
        camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t>,
         RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec>,
         RAJA::expt::LoopPolicy<RAJA::loop_exec>>>;
#endif

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)
using Cuda_launch_policies = camp::list<
         seq_cuda_policies
         , seq_cuda_explicit_policies
#if defined(RAJA_ENABLE_OPENMP)
         , omp_cuda_policies
         , omp_cuda_explicit_policies
#endif
        >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)
using Hip_launch_policies = camp::list<
         seq_hip_policies
#if defined(RAJA_ENABLE_OPENMP)
         , omp_hip_policies
#endif
        >;
#endif // RAJA_ENABLE_HIP


#endif  // __RAJA_test_teams_runtime_execpol_HPP__
