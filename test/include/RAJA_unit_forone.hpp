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
/// Header file of macro for-one CUDA unit tests.
/// Use as:
/// forone<<<1,1>>>( [=] __device__ () {} );
///

#if defined(RAJA_ENABLE_CUDA)

#define RAJA_TEST_ENABLE_GPU
#define RAJA_TEST_DEVICE_LAMBDA [=] __device__

template <typename L>
__global__ void forone (L run)
{
  run();
}

template <typename L>
__global__ void forone_cuda(L run)
{
  run();
}

#elif defined(RAJA_ENABLE_HIP)

#define RAJA_TEST_ENABLE_GPU
#define RAJA_TEST_DEVICE_LAMBDA [=] __device__

template <typename L>
__global__ void forone_hip(L run)
{
  run();
}

#endif


///
/// Header file of macro for-one gpu (cuda or hip) unit tests.
/// Use as:
/// forone<<<1,1>>>( [=] __device__ () {} );
///
template <typename L>
inline void forone_gpu(L&& run)
{
#if defined(RAJA_ENABLE_CUDA)
   forone_cuda<<<1,1>>>(std::forward<L>(run));
   cudaErrchk(cudaGetLastError());
   cudaErrchk(cudaDeviceSynchronize());
#elif defined(RAJA_ENABLE_HIP)
   forone_hip<<<1,1>>>(std::forward<L>(run));
   hipErrchk(hipGetLastError());
   hipErrchk(hipDeviceSynchronize());
#else
   static_assert(std::is_same<L, void>::value,
                 "Not compiled with a GPU");
#endif
}

#endif // RAJA_unit_forone_HPP
