//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout launch tests
//

#ifndef __RAJA_TEST_LAUNCH_DIRECT_UNCHECKED_TEAMS_THREADS_1D_EXECPOL_HPP__
#define __RAJA_TEST_LAUNCH_DIRECT_UNCHECKED_TEAMS_THREADS_1D_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"


#if defined(RAJA_ENABLE_CUDA)

using cuda_direct_unchecked_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>,
             RAJA::LoopPolicy<RAJA::cuda_block_x_direct_unchecked>,
             RAJA::LoopPolicy<RAJA::cuda_thread_x_direct_unchecked>
            >;

using cuda_direct_unchecked_explicit_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::policy::cuda::cuda_launch_explicit_t<true, 0, 0>>,
             RAJA::LoopPolicy<RAJA::cuda_block_x_direct_unchecked>,
             RAJA::LoopPolicy<RAJA::cuda_thread_x_direct_unchecked>
           >;

using Cuda_launch_policies =
  camp::list<
             cuda_direct_unchecked_policies,
             cuda_direct_unchecked_explicit_policies
            >;
#endif  // RAJA_ENABLE_CUDA

#if defined(RAJA_ENABLE_HIP)

using hip_direct_unchecked_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::hip_launch_t<true>>,
             RAJA::LoopPolicy<RAJA::hip_block_x_direct_unchecked>,
             RAJA::LoopPolicy<RAJA::hip_thread_x_direct_unchecked>
           >;

using Hip_launch_policies = camp::list<hip_direct_unchecked_policies>;

#endif // RAJA_ENABLE_HIP


#endif  // __RAJA_TEST_LAUNCH_DIRECT_UNCHECKED_TEAMS_THREADS_1D_EXECPOL_HPP__
