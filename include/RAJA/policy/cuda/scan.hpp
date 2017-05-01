/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_cuda_HXX
#define RAJA_scan_cuda_HXX

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/policy/cuda/policy.hpp"

#include <iterator>
#include <type_traits>

#if defined(RAJA_ENABLE_CUB)
#include "cub/util_allocator.cuh"
#include "cub/device/device_scan.cuh"
#else
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#endif

namespace RAJA
{
namespace impl
{
namespace scan
{

#if defined(RAJA_ENABLE_CUB)
RAJA_INLINE ::cub::CachingDeviceAllocator& getAllocator() {
  static ::cub::CachingDeviceAllocator allocator(true);
  return allocator;
}
#endif

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename Function>
void inclusive_inplace(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op)
{
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, begin, binary_op, len));
  // Allocate temporary storage
  cudaErrchk(getAllocator().DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  // Run
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, begin, binary_op, len));
  // Free temporary storage
  cudaErrchk(getAllocator().DeviceFree(d_temp_storage));
#else
  ::thrust::inclusive_scan(::thrust::device, begin, end, begin, binary_op);
#endif
  RAJA_CUDA_CHECK_AND_SYNC(Async);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename Function,
          typename T>
void exclusive_inplace(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op,
                       T init)
{
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, begin, begin, binary_op, init, len));
  // Allocate temporary storage
  cudaErrchk(getAllocator().DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  // Run
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, begin, begin, binary_op, init, len));
  // Free temporary storage
  cudaErrchk(getAllocator().DeviceFree(d_temp_storage));
#else
  ::thrust::exclusive_scan(
      ::thrust::device, begin, end, begin, init, binary_op);
#endif
  RAJA_CUDA_CHECK_AND_SYNC(Async);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function>
void inclusive(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op)
{
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, out, binary_op, len));
  // Allocate temporary storage
  cudaErrchk(getAllocator().DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  // Run
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, out, binary_op, len));
  // Free temporary storage
  cudaErrchk(getAllocator().DeviceFree(d_temp_storage));
#else
  ::thrust::inclusive_scan(::thrust::device, begin, end, out, binary_op);
#endif
  RAJA_CUDA_CHECK_AND_SYNC(Async);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function,
          typename T>
void exclusive(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op,
               T init)
{
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, begin, out, binary_op, init, len));
  // Allocate temporary storage
  cudaErrchk(getAllocator().DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  // Run
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, begin, out, binary_op, init, len));
  // Free temporary storage
  cudaErrchk(getAllocator().DeviceFree(d_temp_storage));
#else
  ::thrust::exclusive_scan(::thrust::device, begin, end, out, init, binary_op);
#endif
  RAJA_CUDA_CHECK_AND_SYNC(Async);
}

}  // closing brace for scan namespace

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
