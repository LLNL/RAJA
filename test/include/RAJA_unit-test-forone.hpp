//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Header defining "for one" unit test utility so that constructs can be
// tested outside of standard RAJA kernel launch utilities (forall, kernel).
//

#ifndef __RAJA_test_forone_HPP__
#define __RAJA_test_forone_HPP__

#include "RAJA_unit-test-policy.hpp"

///
/// forone<test_policy>( [=] RAJA_HOST_DEVICE(){ /* code to test */ } );
///
template < typename test_policy, typename L >
inline void forone(L&& run);

// test_seq implementation
template < typename L >
inline void forone(test_seq, L&& run)
{
  std::forward<L>(run)();
}

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// test_openmp_target implementation
template < typename L >
inline void forone(test_openmp_target, L&& run)
{
#pragma omp target
  run();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

template <typename L>
__global__ void forone_cuda_global(L run)
{
  run();
}

// test_cuda implementation
template < typename L >
inline void forone(test_cuda, L&& run)
{
   forone_cuda_global<<<1,1>>>(std::forward<L>(run));
   cudaErrchk(cudaGetLastError);
   cudaErrchk(cudaDeviceSynchronize);
}

#endif

#if defined(RAJA_ENABLE_HIP)

template <typename L>
__global__ void forone_hip_global(L run)
{
  run();
}

// test_hip implementation
template < typename L >
inline void forone(test_hip, L&& run)
{
   hipLaunchKernelGGL(forone_hip_global<camp::decay<L>>, dim3(1), dim3(1), 0, 0, std::forward<L>(run));
   hipErrchk(hipGetLastError);
   hipErrchk(hipDeviceSynchronize);
}

#endif

template < typename test_policy, typename L >
void forone(L&& run)
{
  forone(test_policy{}, std::forward<L>(run));
}

#endif // RAJA_test_forone_HPP__
