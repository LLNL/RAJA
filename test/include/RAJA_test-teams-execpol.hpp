//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout teams tests
//

#ifndef __RAJA_test_teams_execpol_HPP__
#define __RAJA_test_teams_execpol_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//Launch policies
using seq_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>,
  RAJA::expt::LoopPolicy<RAJA::loop_exec>
  >;

using Sequential_launch_policies = camp::list<
  seq_policies
  >;

#if defined(RAJA_ENABLE_OPENMP)
using omp_policies = camp::list<
         RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t>,
         RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec>  
  >;

using OpenMP_launch_policies = camp::list<
  omp_policies
  >;

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)

using cuda_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::cuda_launch_t<true>>,
  RAJA::expt::LoopPolicy<RAJA::expt::cuda_global_thread_x>>;

using cuda_explicit_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::policy::cuda::expt::cuda_launch_explicit_t<true, 0, 0>>,
  RAJA::expt::LoopPolicy<RAJA::expt::cuda_global_thread_x>>;

using Cuda_launch_policies = camp::list<
        cuda_policies,
        cuda_explicit_policies
         >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)

using hip_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<true>>,
  RAJA::expt::LoopPolicy<RAJA::expt::cuda_global_thread_x>>;

using hip_explicit_policies = camp::list<
  RAJA::expt::LaunchPolicy<RAJA::policy::hip::expt::hip_launch_explicit_t<true, 0, 0>>,
  RAJA::expt::LoopPolicy<RAJA::expt::cuda_global_thread_x>;

using Hip_launch_policies = camp::list<
      hip_policies,
      hip_explicit_policies
       >;
#endif // RAJA_ENABLE_HIP


#endif  // __RAJA_test_teams_execpol_HPP__
