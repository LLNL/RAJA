//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_unit_forone_HPP
#define RAJA_unit_forone_HPP

#include <RAJA/RAJA.hpp>

#include <type_traits>

///
/// forone_gpu<forone_policy>( RAJA_TEST_DEVICE_LAMBDA(){ /* code to test */ } );
///
template < typename forone_policy, typename L >
inline void forone_gpu(L&& run);

// sequential forone policy
struct forone_seq  { };

// struct with specializations containing information about forone policies
template < typename forone_policy >
struct forone_policy_info;

// alias for equivalent RAJA exec policy to given forone policy
template < typename forone_policy >
using forone_equivalent_exec_policy = typename forone_policy_info<forone_policy>::type;


// forone_seq policy information
template < >
struct forone_policy_info<forone_seq>
{
  using type = RAJA::loop_exec;
  static const char* name() { return "forone_seq"; }
};

// forone_seq implementation
template < typename L >
inline void forone_gpu(forone_seq, L&& run)
{
  std::forward<L>(run)();
}

#if defined(RAJA_ENABLE_CUDA)

#define RAJA_TEST_DEVICE_LAMBDA [=] __device__

// cuda forone policy
struct forone_cuda { };

// forone_cuda policy information
template < >
struct forone_policy_info<forone_cuda>
{
  using type = RAJA::cuda_exec<1>;
  static const char* name() { return "forone_cuda"; }
};

template <typename L>
__global__ void forone (L run)
{
  run();
}

template <typename L>
__global__ void forone_cuda_global(L run)
{
  run();
}

// forone_cuda implementation
template < typename L >
inline void forone_gpu(forone_cuda, L&& run)
{
   forone_cuda_global<<<1,1>>>(std::forward<L>(run));
   cudaErrchk(cudaGetLastError());
   cudaErrchk(cudaDeviceSynchronize());
}

#elif defined(RAJA_ENABLE_HIP)

#define RAJA_TEST_DEVICE_LAMBDA [=] __device__

// hip forone policy
struct forone_hip  { };

// forone_hip policy information
template < >
struct forone_policy_info<forone_hip>
{
  using type = RAJA::hip_exec<1>;
  static const char* name() { return "forone_hip"; }
};

template <typename L>
__global__ void forone_hip_global(L run)
{
  run();
}

// forone_hip implementation
template < typename L >
inline void forone_gpu(forone_hip, L&& run)
{
   hipLaunchKernelGGL(forone_hip_global<camp::decay<L>>, dim3(1), dim3(1), 0, 0, std::forward<L>(run));
   hipErrchk(hipGetLastError());
   hipErrchk(hipDeviceSynchronize());
}

#endif

template < typename forone_policy, typename L >
void forone_gpu(L&& run)
{
  forone_gpu(forone_policy{}, std::forward<L>(run));
}

#endif // RAJA_unit_forone_HPP
