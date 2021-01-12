/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via HIP kernel launch.
 *
 *          These methods should work on any platform that supports
 *          HIP devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_hip_HPP
#define RAJA_forall_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <algorithm>
#include "hip/hip_runtime.h"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

namespace RAJA
{

namespace policy
{
namespace hip
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
hip_dim_t getGridDim(hip_dim_member_t len, hip_dim_t blockDim)
{
  hip_dim_member_t block_size = blockDim.x * blockDim.y * blockDim.z;

  hip_dim_member_t gridSize = (len + block_size - 1) / block_size;

  return gridSize;
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 1D grid of 1D blocks
 *
 ******************************************************************************
 */
__device__ __forceinline__ unsigned int getGlobalIdx_1D_1D()
{
  unsigned int blockId = blockIdx.x;
  unsigned int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned int getGlobalNumThreads_1D_1D()
{
  unsigned int numThreads = blockDim.x * gridDim.x;
  return numThreads;
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 3D grid of 3D blocks
 *
 ******************************************************************************
 */
__device__ __forceinline__ unsigned int getGlobalIdx_3D_3D()
{
  unsigned int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  unsigned int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                          (threadIdx.z * (blockDim.x * blockDim.y)) +
                          (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned int getGlobalNumThreads_3D_3D()
{
  unsigned int numThreads =
      blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
  return numThreads;
}

//
//////////////////////////////////////////////////////////////////////
//
// HIP kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  HIP kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
__launch_bounds__(BlockSize, 1) __global__
    void forall_hip_kernel(LOOP_BODY loop_body,
                            const Iterator idx,
                            IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  if (ii < length) {
    body(idx[ii]);
  }
}

}  // namespace impl

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for HIP execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE resources::EventProxy<resources::Hip> forall_impl(resources::Hip &hip_res,
                                                    hip_exec<BlockSize, Async>,
                                                    Iterable&& iter,
                                                    LoopBody&& loop_body)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  auto func = impl::forall_hip_kernel<BlockSize, Iterator, LOOP_BODY, IndexType>;

  hipStream_t stream = hip_res.get_stream();

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
    hip_dim_t blockSize{BlockSize, 1, 1};
    hip_dim_t gridSize = impl::getGridDim(static_cast<hip_dim_member_t>(len), blockSize);

    RAJA_FT_BEGIN;

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //  printf("gridsize = (%d,%d), blocksize = %d\n",
    //         (int)gridSize.x,
    //         (int)gridSize.y,
    //         (int)blockSize.x);

    {
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::hip::make_launch_body(
          gridSize, blockSize, shmem, stream, std::forward<LoopBody>(loop_body));


      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::hip::launch((const void*)func, gridSize, BlockSize, args, shmem, stream);
    }

    if (!Async) { RAJA::hip::synchronize(stream); }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Hip>(&hip_res);
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as HIP kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         HIP execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_impl(ExecPolicy<seq_segit, hip_exec<BlockSize, Async>>,
                             const TypedIndexSet<SegmentTypes...>& iset,
                             LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     detail::CallForall(),
                     hip_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::hip::synchronize();
}

}  // namespace hip

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
