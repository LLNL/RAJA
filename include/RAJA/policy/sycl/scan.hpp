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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_sycl_HPP
#define RAJA_scan_sycl_HPP

#include "RAJA/config.hpp"
#include "camp/resource/sycl.hpp"

#if defined(RAJA_ENABLE_SYCL)


#include <iterator>

// #include <numeric>

// #include <type_traits>

#include "RAJA/pattern/detail/algorithm.hpp"

// #include "RAJA/policy/sycl/MemUtils_SYCL.hpp"

#include "RAJA/policy/sycl/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{

template <size_t BLOCK_SIZE, bool Async, typename InputIter, typename Function>
RAJA_INLINE 
camp::resources::EventProxy<camp::resources::Sycl> 
inclusive_inplace(
    camp::resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async>,
    InputIter begin,
    InputIter end,
    Function binary_op)
{
    ::sycl::queue* sycl_queue = sycl_res.get_queue();
 
    using valueT = typename std::remove_reference<decltype(*begin)>::type;

     // Calculate the size of the input range
    size_t size = std::distance(begin, end);

    ::sycl::buffer<valueT, 1> buf(begin, ::sycl::range<1>(size));

    // Submit the kernel to the SYCL queue
    sycl_queue->submit([&](::sycl::handler& cgh) {
        // ::sycl::accessor<valueT, 1, ::sycl::access::mode::read_write
        cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> idx) {
            if (idx[0] != 0) {
                *(begin + idx[0]) = binary_op(*(begin + idx[0] - 1), *(begin + idx[0]));
            }
        });
    });

  sycl_res.wait();
  return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename Function,
          typename TT>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
exclusive_inplace(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    Function binary_op,
    TT initVal)
{

    ::sycl::queue* sycl_queue = sycl_res.get_queue();

     // Calculate the size of the input range
    size_t size = std::distance(begin, end);
    using valueT = typename std::remove_reference<decltype(*begin)>::type;
    valueT agg = *begin;
    // Submit the kernel to the SYCL queue
    sycl_queue->submit([&](::sycl::handler& cgh) {

        cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> idx) {
            if (idx[0] == 0) {
                *begin = binary_op(*begin, initVal);
            } else {
                // agg = binary_op(agg, begin + idx[0]);
                *(begin + idx[0])  = binary_op(*(begin + idx[0] - 1), *(begin + idx[0]));
            }
        });
    });

    sycl_res.wait();
    return camp::resources::EventProxy<camp::resources::Sycl>(sycl_res);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
inclusive(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op)
{
  // ::sycl::joint_inclusive_scan()
    using std::distance;
    std::copy(begin, end, out);
    return inclusive_inplace(sycl_res, exec, out, out + distance(begin, end), binary_op);
}

template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function,
          typename TT>
RAJA_INLINE 
resources::EventProxy<resources::Sycl> 
exclusive(
    resources::Sycl sycl_res,
    ::RAJA::policy::sycl::sycl_exec<BLOCK_SIZE, Async> exec,
    InputIter begin,
    InputIter end,
    OutputIter out,
    Function binary_op,
    TT initVal)
{
    using std::distance;
    std::copy(begin, end, out);
    return exclusive_inplace(sycl_res, exec, out, out + distance(begin, end), binary_op, initVal);
}

}  // namespace scan
}  // namespace impl
}  // namespace RAJA

#endif  // closing endif for RAJA enable Sycl guard

#endif  // closing endif for header include guard