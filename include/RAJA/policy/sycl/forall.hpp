/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for SYCL.
 *
 *          These methods should work on any platform that supports SYCL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_sycl_HPP
#define RAJA_forall_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "CL/sycl.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/policy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/forall.hpp"


namespace RAJA
{

namespace policy
{
namespace sycl
{

namespace impl
{

/*!
 ******************************************************************************
 *
 * \brief calculate gridDim from length of iteration and blockDim
 *
 ******************************************************************************
 */
RAJA_INLINE
sycl_dim_t getGridDim(size_t len, sycl_dim_t block_size)
{
  sycl_dim_t gridSize{block_size * ((len + block_size - 1) / block_size)};

  return gridSize;
}

}  // namespace impl


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for SYCL execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE void forall_impl(sycl_exec<BlockSize, Async>,
                             Iterable&& iter,
                             LoopBody&& loop_body)
{

  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);
  IndexType offset = *begin;

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0) {

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize{BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);

    cl::sycl::queue q = ::RAJA::sycl::detail::getQueue();

    // Non-trivially copyable
    Iterator* idx = (Iterator*) cl::sycl::malloc_device(sizeof(Iterator), q);
    auto e = q.memcpy(idx, &begin, sizeof(Iterator));
    e.wait();

    LOOP_BODY* lbody = (LOOP_BODY*) cl::sycl::malloc_device(sizeof(LOOP_BODY), q);
    auto e2 = q.memcpy(lbody, &loop_body, sizeof(LOOP_BODY));
    e2.wait();

    q.submit([&](cl::sycl::handler& h) {
      h.parallel_for( cl::sycl::nd_range<1>{gridSize, blockSize},
                      [=] (cl::sycl::nd_item<1> it) {

        using RAJA::internal::thread_privatize;
        auto privatizer = thread_privatize(*lbody);
        auto& body = privatizer.get_priv();

        size_t ii = it.get_global_id(0);

        if (ii < len) {
          body((*idx)[ii]);
        }
      });
    });

    if (!Async) { q.wait(); }

    cl::sycl::free(idx, q);
    cl::sycl::free(lbody, q);

  }
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as SYCL kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         SYCL execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_impl(ExecPolicy<seq_segit, sycl_exec<BlockSize, Async>>,
                             const TypedIndexSet<SegmentTypes...>& iset,
                             LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     detail::CallForall(),
                     sycl_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) {
    cl::sycl::queue q = ::RAJA::sycl::detail::getQueue();
    q.wait();
  };
}

}  // namespace sycl
}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_SYCL)

#endif  // closing endif for header file include guard
