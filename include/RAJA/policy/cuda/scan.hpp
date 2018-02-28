/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_cuda_HPP
#define RAJA_scan_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include <iterator>
#include <type_traits>

#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

namespace RAJA
{
namespace impl
{
namespace scan
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <size_t BLOCK_SIZE, bool Async, typename InputIter, typename Function>
void inclusive_inplace(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op)
{
  cudaStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  cudaErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              len,
                                              stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
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
  cudaStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
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
  cudaStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  cudaErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
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
  cudaStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              out,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  cudaErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              out,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
}

}  // closing brace for scan namespace

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
