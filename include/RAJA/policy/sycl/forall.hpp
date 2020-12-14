/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via SYCL kernel launch.
 *
 *          These methods should work on any platform that supports
 *          SYCL devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_sycl_HPP
#define RAJA_forall_sycl_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
#include <chrono>

#include "RAJA/pattern/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"
//#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

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
cl::sycl::range<1> getGridDim(size_t len, size_t block_size)
{
  size_t size = {block_size * ((len + block_size - 1) / block_size)};
  cl::sycl::range<1> gridSize(size);
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
RAJA_INLINE resources::EventProxy<resources::Sycl>  forall_impl(resources::Sycl &sycl_res,
                                                                sycl_exec<BlockSize, Async>,
                                                                Iterable&& iter,
                                                                LoopBody&& loop_body)
//RAJA_INLINE void launchSyclTrivial(size_t BlockSize, bool Async, Iterable&& iter, LoopBody&& loop_body)
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

std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0) {

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize{BlockSize};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<size_t>(len), BlockSize);

    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();
    std::cout << "RAJA launch qu ptr = " << q << std::endl;

    q->submit([&](cl::sycl::handler& h) {

      h.parallel_for( cl::sycl::nd_range<1>{gridSize, blockSize},
                      [=]  (cl::sycl::nd_item<1> it) {

        size_t ii = it.get_global_id(0);

        if (ii < len) {
          loop_body(begin[ii]);
        }
      });
    });

    if (!Async) { q->wait(); }
  }
std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();

std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << "[µs]" << std::endl;
std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (stop - start).count() << "[ns]" << std::endl;

  return resources::EventProxy<resources::Sycl>(&sycl_res);
}
/*
template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE resources::EventProxy<resources::Sycl> forall_impl(resources::Sycl &sycl_res,
                                                    sycl_exec<BlockSize, Async>,
                                                    Iterable&& iter,
                                                    LoopBody&& loop_body)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

//  auto func = impl::forall_cuda_kernel<BlockSize, Iterator, LOOP_BODY, IndexType>;

//  cudaStream_t stream = cuda_res.get_stream();

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0) {

    //
    // Compute the number of blocks
    //
    sycl_dim_t blockSize{BlockSize, 1, 1};
    sycl_dim_t gridSize = impl::getGridDim(static_cast<sycl_dim_member_t>(len), blockSize);

    RAJA_FT_BEGIN;

    //
    // Setup shared memory buffers
    //
//    size_t shmem = 0;

    //  printf("gridsize = (%d,%d), blocksize = %d\n",
    //         (int)gridSize.x,
    //         (int)gridSize.y,
    //         (int)blockSize.x);
// TODO: BRIAN
    {
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          gridSize, blockSize, shmem, stream, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::cuda::launch((const void*)func, gridSize, blockSize, args, shmem, stream);
    }

    if (!Async) { RAJA::cuda::synchronize(stream); }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(&cuda_res);
}
*/

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
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();
    q->wait();
  };
}


template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Sycl> forall_impl(resources::Sycl &r,
                                                    ExecPolicy<seq_segit, sycl_exec<BlockSize, Async>>,
                                                    const TypedIndexSet<SegmentTypes...>& iset,
                                                    LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     sycl_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) {
    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue(); 
    q->wait();
  }

  return resources::EventProxy<resources::Sycl>(&r);
}

}  // namespace cuda

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
