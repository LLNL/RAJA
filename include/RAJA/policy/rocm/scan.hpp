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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_scan_rocm_HPP
#define RAJA_scan_rocm_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/policy/rocm/policy.hpp"

#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"

#include <iterator>
#include <type_traits>

//#if defined(RAJA_ENABLE_CUB)
//#include "cub/device/device_scan.cuh"
//#include "cub/util_allocator.cuh"
//#else
//#include <thrust/device_ptr.h>
//#include <thrust/execution_policy.h>
//#include <thrust/functional.h>
//#include <thrust/scan.h>
//#include <thrust/system/rocm/execution_policy.h>
//#endif

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
void inclusive_inplace(const ::RAJA::rocm_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op)
{
  rocmStream_t stream = 0;
#if 0
// need to implement scan, cant use CUB or thrust
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  rocmErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      rocm::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  rocmErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              len,
                                              stream));
  // Free temporary storage
  rocm::device_mempool_type::getInstance().free(d_temp_storage);
#else
  ::thrust::inclusive_scan(
      ::thrust::rocm::par.on(stream), begin, end, begin, binary_op);
#endif
  rocm::launch(stream);
  if (!Async) rocm::synchronize(stream);
#endif
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
void exclusive_inplace(const ::RAJA::rocm_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op,
                       T init)
{
  rocmStream_t stream = 0;
#if 0
// need to implement scan, cant use CUB or thrust
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  rocmErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      rocm::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  rocmErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Free temporary storage
  rocm::device_mempool_type::getInstance().free(d_temp_storage);
#else
  ::thrust::exclusive_scan(
      ::thrust::rocm::par.on(stream), begin, end, begin, init, binary_op);
#endif
  rocm::launch(stream);
  if (!Async) rocm::synchronize(stream);
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
void inclusive(const ::RAJA::rocm_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op)
{
  rocmStream_t stream = 0;
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  rocmErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
  // Allocate temporary storage
  d_temp_storage =
      rocm::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  rocmErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
  // Free temporary storage
  rocm::device_mempool_type::getInstance().free(d_temp_storage);
#else
  ::thrust::inclusive_scan(
      ::thrust::rocm::par.on(stream), begin, end, out, binary_op);
#endif
  rocm::launch(stream);
  if (!Async) rocm::synchronize(stream);
#endif
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
void exclusive(const ::RAJA::rocm_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op,
               T init)
{
  rocmStream_t stream = 0;
#if 0
// need to implement scan, cant use CUB or thrust
#if defined(RAJA_ENABLE_CUB)
  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  rocmErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              out,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      rocm::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
  rocmErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              out,
                                              binary_op,
                                              init,
                                              len,
                                              stream));
  // Free temporary storage
  rocm::device_mempool_type::getInstance().free(d_temp_storage);
#else
  ::thrust::exclusive_scan(
      ::thrust::rocm::par.on(stream), begin, end, out, init, binary_op);
#endif
  rocm::launch(stream);
  if (!Async) rocm::synchronize(stream);
#endif
}

}  // closing brace for scan namespace

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
