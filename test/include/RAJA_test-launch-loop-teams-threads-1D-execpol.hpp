//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout launch tests
//

#ifndef __RAJA_TEST_LAUNCH_TEAMS_THREADS_LOOP_3D_EXECPOL_HPP__
#define __RAJA_TEST_LAUNCH_TEAMS_THREADS_LOOP_3D_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//Launch policies
using seq_policies = camp::list<
  RAJA::LaunchPolicy<RAJA::seq_launch_t>,
  RAJA::LoopPolicy<RAJA::loop_exec>,
  RAJA::LoopPolicy<RAJA::loop_exec>
  >;

using Sequential_launch_policies = camp::list<
  seq_policies
  >;

#if defined(RAJA_ENABLE_OPENMP)
using omp_policies = camp::list<
         RAJA::LaunchPolicy<RAJA::omp_launch_t>,
         RAJA::LoopPolicy<RAJA::omp_for_exec>,  
         RAJA::LoopPolicy<RAJA::loop_exec>
  >;

using OpenMP_launch_policies = camp::list<
  omp_policies
  >;

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)

using cuda_loop_policies = camp::list<
  RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>,
  RAJA::LoopPolicy<RAJA::cuda_block_x_loop>,
  RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>
  >;

using cuda_loop_explicit_policies = camp::list<
  RAJA::LaunchPolicy<RAJA::policy::cuda::cuda_launch_explicit_t<true, 0, 0>>,
  RAJA::LoopPolicy<RAJA::cuda_block_x_loop>,
  RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>
  >;

using Cuda_launch_policies = camp::list<
  cuda_loop_policies,
  cuda_loop_explicit_policies
  >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)

using hip_loop_policies = camp::list<
  RAJA::LaunchPolicy<RAJA::hip_launch_t<true>>,
  RAJA::LoopPolicy<RAJA::hip_block_x_loop>,
  RAJA::LoopPolicy<RAJA::hip_thread_x_loop>
  >;

using Hip_launch_policies = camp::list<
      hip_loop_policies
       >;
#endif // RAJA_ENABLE_HIP

#if defined(RAJA_ENABLE_SYCL)
using sycl_loop_policies = camp::list<
  RAJA::LaunchPolicy<RAJA::sycl_launch_t<true>>,
  RAJA::LoopPolicy<RAJA::sycl_group_0_loop>,
  RAJA::LoopPolicy<RAJA::sycl_local_0_loop>
  >;

using Sycl_launch_policies = camp::list<  
  sycl_loop_policies
  >;
#endif


#endif  // __RAJA_test_launch_teams_threads_loop_3D_execpol_HPP__
