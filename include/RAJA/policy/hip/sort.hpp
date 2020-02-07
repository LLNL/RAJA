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

#ifndef RAJA_sort_hip_HPP
#define RAJA_sort_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <climits>
#include <iterator>
#include <type_traits>

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_radix_sort.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_radix_sort.cuh"
#endif

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{

#if defined(__HIPCC__)
  template < typename R >
  using double_buffer = ::rocprim::double_buffer<R>;
#elif defined(__CUDACC__)
  template < typename R >
  using double_buffer = ::cub::DoubleBuffer<R>;
#endif

}

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
stable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
     Iter,
     Iter,
     Compare)
{
  static_assert(concepts::all_of<
                  type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                  std::is_pointer<Iter>,
                  concepts::any_of<
                    camp::is_same<Compare, operators::less<RAJA::detail::IterVal<Iter>>>,
                    camp::is_same<Compare, operators::greater<RAJA::detail::IterVal<Iter>>>>>::value,
                "RAJA stable_sort<hip_exec> is only implemented for pointers to arithmetic types and RAJA::operators::less and RAJA::operators::greater.");
}

/*!
        \brief stable sort given range in ascending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
stable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
     Iter begin,
     Iter end,
     operators::less<RAJA::detail::IterVal<Iter>>)
{
  hipStream_t stream = 0;

  using R = RAJA::detail::IterVal<Iter>;

  int len = std::distance(begin, end);
  int begin_bit=0;
  int end_bit=sizeof(R)*CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = hip::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  detail::double_buffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  hipErrchk(::rocprim::radix_sort_keys(d_temp_storage,
                                       temp_storage_bytes,
                                       d_keys,
                                       len,
                                       begin_bit,
                                       end_bit,
                                       stream));
#elif defined(__CUDACC__)
  cudaErrchk(::cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys,
                                              len,
                                              begin_bit,
                                              end_bit,
                                              stream));
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  hipErrchk(::rocprim::radix_sort_keys(d_temp_storage,
                                       temp_storage_bytes,
                                       d_keys,
                                       len,
                                       begin_bit,
                                       end_bit,
                                       stream));
#elif defined(__CUDACC__)
  cudaErrchk(::cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                              temp_storage_bytes,
                                              d_keys,
                                              len,
                                              begin_bit,
                                              end_bit,
                                              stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_out) {

    // copy
    hipErrchk(hipMemcpyAsync(begin, d_out, len*sizeof(R), hipMemcpyDefault, stream));
  }

  hip::device_mempool_type::getInstance().free(d_out);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
}

/*!
        \brief stable sort given range in descending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
stable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
     Iter begin,
     Iter end,
     operators::greater<RAJA::detail::IterVal<Iter>>)
{
  hipStream_t stream = 0;

  using R = RAJA::detail::IterVal<Iter>;

  int len = std::distance(begin, end);
  int begin_bit=0;
  int end_bit=sizeof(R)*CHAR_BIT;

  // Allocate temporary storage for the output array
  R* d_out = hip::device_mempool_type::getInstance().malloc<R>(len);

  // use cub double buffer to reduce temporary memory requirements
  // by allowing cub to write to the begin buffer
  detail::double_buffer<R> d_keys(begin, d_out);

  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
  hipErrchk(::rocprim::radix_sort_keys_desc(d_temp_storage,
                                            temp_storage_bytes,
                                            d_keys,
                                            len,
                                            begin_bit,
                                            end_bit,
                                            stream));
#elif defined(__CUDACC__)
  cudaErrchk(::cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        len,
                                                        begin_bit,
                                                        end_bit,
                                                        stream));
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);

  // Run
#if defined(__HIPCC__)
  hipErrchk(::rocprim::radix_sort_keys_desc(d_temp_storage,
                                            temp_storage_bytes,
                                            d_keys,
                                            len,
                                            begin_bit,
                                            end_bit,
                                            stream));
#elif defined(__CUDACC__)
  cudaErrchk(::cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_keys,
                                                        len,
                                                        begin_bit,
                                                        end_bit,
                                                        stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  if (d_keys.Current() == d_out) {

    // copy
    hipErrchk(hipMemcpyAsync(begin, d_out, len*sizeof(R), hipMemcpyDefault, stream));
  }

  hip::device_mempool_type::getInstance().free(d_out);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
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
unstable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
         Iter,
         Iter,
         Compare)
{
  static_assert(concepts::all_of<
                  type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                  std::is_pointer<Iter>,
                  concepts::any_of<
                    camp::is_same<Compare, operators::less<RAJA::detail::IterVal<Iter>>>,
                    camp::is_same<Compare, operators::greater<RAJA::detail::IterVal<Iter>>>>>::value,
                "RAJA sort<hip_exec> is only implemented for pointers to arithmetic types and RAJA::operators::less and RAJA::operators::greater.");
}

/*!
        \brief sort given range in ascending order
*/
template <size_t BLOCK_SIZE, bool Async, typename Iter>
concepts::enable_if<type_traits::is_arithmetic<RAJA::detail::IterVal<Iter>>,
                    std::is_pointer<Iter>>
unstable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>& p,
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
unstable(const ::RAJA::hip_exec<BLOCK_SIZE, Async>& p,
     Iter begin,
     Iter end,
     operators::greater<RAJA::detail::IterVal<Iter>> comp)
{
  stable(p, begin, end, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
