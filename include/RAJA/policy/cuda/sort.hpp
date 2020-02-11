/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_cuda_HPP
#define RAJA_sort_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <climits>
#include <iterator>
#include <type_traits>

#include "cub/device/device_radix_sort.cuh"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

/*!
        \brief static assert unimplemented stable sort
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter, typename Compare>
concepts::enable_if<concepts::negate<concepts::all_of<
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>,
                      concepts::any_of<
                        camp::is_same<Compare, operators::less<RAJA::detail::IterVal<Iter>>>,
                        camp::is_same<Compare, operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
stable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
       Iter,
       Iter,
       Compare)
{
  static_assert (std::is_pointer<Iter>::value,
      "stable_sort<cuda_exec> is only implemented for pointers");
  using iterval = RAJA::detail::IterVal<Iter>;
  static_assert (type_traits::is_arithmetic<iterval>::value,
      "stable_sort<cuda_exec> is only implemented for arithmetic types");
  static_assert (concepts::any_of<
      camp::is_same<Compare, operators::less<iterval>>,
      camp::is_same<Compare, operators::greater<iterval>>>::value,
      "stable_sort<cuda_exec> is only implemented for RAJA::operators::less or RAJA::operators::greater");
}

/*!
        \brief stable sort given range in ascending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
stable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
       Iter begin,
       Iter end,
       operators::less<RAJA::detail::IterVal<Iter>>)
{
  cudaStream_t stream = 0;

  using R = RAJA::detail::IterVal<Iter>;

  int len = std::distance(begin, end);
  int begin_bit=0;
  int end_bit=sizeof(R)*CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = cuda::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  cub::DoubleBuffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys,
                                              len,
                                              begin_bit,
                                              end_bit,
                                              stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  cudaErrchk(::cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys,
                                              len,
                                              begin_bit,
                                              end_bit,
                                              stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_out) {

    // copy
    cudaErrchk(cudaMemcpyAsync(begin, d_out, len*sizeof(R), cudaMemcpyDefault, stream));
  }

  cuda::device_mempool_type::getInstance().free(d_out);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
}

/*!
        \brief stable sort given range in descending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
stable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
       Iter begin,
       Iter end,
       operators::greater<RAJA::detail::IterVal<Iter>>)
{
  cudaStream_t stream = 0;

  using R = RAJA::detail::IterVal<Iter>;

  int len = std::distance(begin, end);
  int begin_bit=0;
  int end_bit=sizeof(R)*CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = cuda::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  cub::DoubleBuffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaErrchk(::cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        len,
                                                        begin_bit,
                                                        end_bit,
                                                        stream));
  // Allocate temporary storage
  d_temp_storage =
      cuda::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
  cudaErrchk(::cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        len,
                                                        begin_bit,
                                                        end_bit,
                                                        stream));
  // Free temporary storage
  cuda::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_out) {

    // copy
    cudaErrchk(cudaMemcpyAsync(begin, d_out, len*sizeof(R), cudaMemcpyDefault, stream));
  }

  cuda::device_mempool_type::getInstance().free(d_out);

  cuda::launch(stream);
  if (!Async) cuda::synchronize(stream);
}


/*!
        \brief static assert unimplemented sort
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter, typename Compare>
concepts::enable_if<concepts::negate<concepts::all_of<
                      type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                      std::is_pointer<Iter>,
                      concepts::any_of<
                        camp::is_same<Compare, operators::less<RAJA::detail::IterVal<Iter>>>,
                        camp::is_same<Compare, operators::greater<RAJA::detail::IterVal<Iter>>>>>>>
unstable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>&,
         Iter,
         Iter,
         Compare)
{
  static_assert (std::is_pointer<Iter>::value,
      "sort<cuda_exec> is only implemented for pointers");
  using iterval = RAJA::detail::IterVal<Iter>;
  static_assert (type_traits::is_arithmetic<iterval>::value,
      "sort<cuda_exec> is only implemented for arithmetic types");
  static_assert (concepts::any_of<
      camp::is_same<Compare, operators::less<iterval>>,
      camp::is_same<Compare, operators::greater<iterval>>>::value,
      "sort<cuda_exec> is only implemented for RAJA::operators::less or RAJA::operators::greater");
}

/*!
        \brief sort given range in ascending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
unstable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>& p,
         Iter begin,
         Iter end,
         operators::less<RAJA::detail::IterVal<Iter>> comp)
{
  stable(p, begin, end, comp);
}

/*!
        \brief sort given range in descending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
unstable(const ::RAJA::cuda_exec<BLOCK_SIZE, Async>& p,
         Iter begin,
         Iter end,
         operators::greater<RAJA::detail::IterVal<Iter>> comp)
{
  stable(p, begin, end, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
