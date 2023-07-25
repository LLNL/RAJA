//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout teams tests
//

#ifndef __RAJA_TEST_LAUNCH_RUNTIME_EXECPOL_HPP__
#define __RAJA_TEST_LAUNCH_RUNTIME_EXECPOL_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

//Launch policies
#if defined(RAJA_ENABLE_CUDA)
using seq_cuda_policies =
  camp::list<
              RAJA::LaunchPolicy<RAJA::seq_launch_t,RAJA::cuda_launch_t<true>>,
              RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_direct>,
              RAJA::LoopPolicy<RAJA::seq_exec,RAJA::cuda_thread_x_loop>
            >;

using seq_cuda_explicit_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::seq_launch_t,RAJA::policy::cuda::cuda_launch_explicit_t<true, 0, 0>>,
             RAJA::LoopPolicy<RAJA::seq_exec, RAJA::cuda_block_x_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::cuda_thread_x_loop>
            >;

using Sequential_launch_policies =
  camp::list<
             seq_cuda_policies,
             seq_cuda_explicit_policies
            >;

#elif defined(RAJA_ENABLE_HIP)
using seq_hip_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::seq_launch_t,RAJA::hip_launch_t<true>>,
             RAJA::LoopPolicy<RAJA::seq_exec, RAJA::hip_block_x_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::hip_thread_x_loop>
            >;

using Sequential_launch_policies = camp::list<seq_hip_policies>;

#elif defined(RAJA_ENABLE_SYCL)

using seq_sycl_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::seq_launch_t,RAJA::sycl_launch_t<true>>,
             RAJA::LoopPolicy<RAJA::seq_exec, RAJA::sycl_group_0_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::sycl_local_0_loop>
            >;

using Sequential_launch_policies = camp::list<seq_sycl_policies>;

#else
using Sequential_launch_policies =
  camp::list<
    camp::list<
               RAJA::LaunchPolicy<RAJA::seq_launch_t>,
               RAJA::LoopPolicy<RAJA::seq_exec>,
               RAJA::LoopPolicy<RAJA::seq_exec>
              >
            >;
#endif // Sequential


#if defined(RAJA_ENABLE_OPENMP)

#if defined(RAJA_ENABLE_CUDA)

using omp_cuda_policies =
  camp::list<
              RAJA::LaunchPolicy<RAJA::omp_launch_t,RAJA::cuda_launch_t<false>>,
              RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::cuda_block_x_direct>,
              RAJA::LoopPolicy<RAJA::seq_exec,RAJA::cuda_thread_x_loop>
            >;

using omp_cuda_explicit_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::omp_launch_t,RAJA::policy::cuda::cuda_launch_explicit_t<false, 0, 0>>,
             RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::cuda_block_x_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::cuda_thread_x_loop>
            >;

using OpenMP_launch_policies =
  camp::list<
             omp_cuda_policies,
             omp_cuda_explicit_policies
            >;

#elif defined(RAJA_ENABLE_HIP)

using omp_hip_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::omp_launch_t,RAJA::hip_launch_t<false>>,
             RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::hip_block_x_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::hip_thread_x_loop>
            >;

using OpenMP_launch_policies = camp::list<omp_hip_policies>;

#elif defined(RAJA_ENABLE_SYCL)

using omp_sycl_policies =
  camp::list<
             RAJA::LaunchPolicy<RAJA::omp_launch_t,RAJA::sycl_launch_t<false>>,
             RAJA::LoopPolicy<RAJA::omp_for_exec, RAJA::sycl_group_0_direct>,
             RAJA::LoopPolicy<RAJA::seq_exec,RAJA::sycl_local_0_loop>
            >;

using OpenMP_launch_policies = camp::list<omp_sycl_policies>;

#else

using OpenMP_launch_policies =
  camp::list<
    camp::list<
                RAJA::LaunchPolicy<RAJA::omp_launch_t>,
                RAJA::LoopPolicy<RAJA::omp_parallel_for_exec>,
                RAJA::LoopPolicy<RAJA::seq_exec>
               >
             >;
#endif

#endif  // RAJA_ENABLE_OPENMP

#if defined(RAJA_ENABLE_CUDA)

using Cuda_launch_policies =
  camp::list<
             seq_cuda_policies
            ,seq_cuda_explicit_policies

#if defined(RAJA_ENABLE_OPENMP)
            ,omp_cuda_policies
            ,omp_cuda_explicit_policies
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

#if defined(RAJA_ENABLE_SYCL)

using Sycl_launch_policies = camp::list<
         seq_sycl_policies

#if defined(RAJA_ENABLE_OPENMP)
         , omp_sycl_policies
#endif
        >;

#endif // RAJA_ENABLE_SYCL

#endif  // __RAJA_TEST_LAUNCH_RUNTIME_EXECPOL_HPP__
