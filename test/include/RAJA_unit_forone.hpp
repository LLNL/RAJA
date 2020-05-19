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
/// forone<forone_policy>( [=] RAJA_HOST_DEVICE(){ /* code to test */ } );
///
template < typename forone_policy, typename L >
inline void forone(L&& run);

// base classes to represent host or device in exec_dispatcher
struct RunOnHost {}; 
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
struct RunOnDevice {};
#endif

// sequential forone policy
struct forone_seq : public RunOnHost  { };

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
inline void forone(forone_seq, L&& run)
{
  std::forward<L>(run)();
}

#if defined(RAJA_ENABLE_CUDA)

// cuda forone policy
struct forone_cuda : public RunOnDevice { };

// forone_cuda policy information
template < >
struct forone_policy_info<forone_cuda>
{
  using type = RAJA::cuda_exec<1>;
  static const char* name() { return "forone_cuda"; }
};

template <typename L>
__global__ void forone_cuda_global(L run)
{
  run();
}

// forone_cuda implementation
template < typename L >
inline void forone(forone_cuda, L&& run)
{
   forone_cuda_global<<<1,1>>>(std::forward<L>(run));
   cudaErrchk(cudaGetLastError());
   cudaErrchk(cudaDeviceSynchronize());
}

#elif defined(RAJA_ENABLE_HIP)

// hip forone policy
struct forone_hip : public RunOnDevice { };

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
inline void forone(forone_hip, L&& run)
{
   hipLaunchKernelGGL(forone_hip_global<camp::decay<L>>, dim3(1), dim3(1), 0, 0, std::forward<L>(run));
   hipErrchk(hipGetLastError());
   hipErrchk(hipDeviceSynchronize());
}

#endif

template < typename forone_policy, typename L >
void forone(L&& run)
{
  forone(forone_policy{}, std::forward<L>(run));
}

#endif // RAJA_unit_forone_HPP
