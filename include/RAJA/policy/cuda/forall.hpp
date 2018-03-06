/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
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

#ifndef RAJA_forall_cuda_HPP
#define RAJA_forall_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/pattern/forall.hpp"

#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include <algorithm>

namespace RAJA
{

namespace policy
{
namespace cuda
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
unsigned getGridDim(size_t len, dim3 blockDim)
{
  size_t block_size = blockDim.x * blockDim.y * blockDim.z;

  size_t gridSize = (len + block_size - 1) / block_size;

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

__device__ __forceinline__ unsigned int getGlobalIdx_1D_1D(unsigned int blockId)
{
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
  unsigned int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                          + (threadIdx.z * (blockDim.x * blockDim.y))
                          + (threadIdx.y * blockDim.x) + threadIdx.x;
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
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernel forall template for indirection array.
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
__launch_bounds__(BlockSize, 1) __global__
    void forall_cuda_kernel(LOOP_BODY loop_body,
                            const Iterator idx,
                            unsigned int num_logical_blocks,
                            IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto &body = privatizer.get_priv();

  unsigned int logical_block = (unsigned int)blockIdx.x;
  while(logical_block < num_logical_blocks){
    auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D(logical_block));
    if (ii < length) {
      body(idx[ii]);
    }

    logical_block += gridDim.x;
  }
}



/*!
 ******************************************************************************
 *
 * \brief  CUDA kernel forall with occupancy calculator
 *
 ******************************************************************************
 */
template <typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
__global__
    void forall_cuda_kernel_occ(LOOP_BODY loop_body,
                            const Iterator idx,
                            unsigned int num_logical_blocks,
                            IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto &body = privatizer.get_priv();

  unsigned int logical_block = (unsigned int)blockIdx.x;
  while(logical_block < num_logical_blocks){
    auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D(logical_block));
    if (ii < length) {
      body(idx[ii]);
    }

    logical_block += gridDim.x;
  }
}

}  // end impl namespace

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t blockSize, bool Async>
RAJA_INLINE void forall_impl(cuda_exec<blockSize, Async>,
                             Iterable&& iter,
                             LoopBody&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);

  auto len = std::distance(begin, end);

  if (len > 0 && blockSize > 0) {

    RAJA_FT_BEGIN;

    cudaStream_t stream = 0;

    // Compute number of logical blocks
    auto num_logical_blocks = impl::getGridDim(len, blockSize);

    // Compute how many blocks will fit on each SM
    int threads_per_sm = RAJA::cuda::device_info::get_max_threads_per_sm();
    int blocks_per_sm = threads_per_sm / blockSize;

    // Now get how many SM's there are, and compute how many blocks we can run
    int num_sm = RAJA::cuda::device_info::get_num_sm();
    int gridSize = num_sm * blocks_per_sm;

    // Restrict actual number of blocks, if it exceeds our iteration space
    gridSize = (num_logical_blocks < gridSize) ? num_logical_blocks : gridSize;

    impl::forall_cuda_kernel<blockSize><<<gridSize, blockSize, 0, stream>>>(
        RAJA::cuda::make_launch_body(
            gridSize, blockSize, 0, stream, std::forward<LoopBody>(loop_body)),
        std::move(begin),
        num_logical_blocks,
        len);
    RAJA::cuda::peekAtLastError();

    RAJA::cuda::launch(stream);
    if (!Async) RAJA::cuda::synchronize(stream);

    RAJA_FT_END;
  }
}



//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, bool Async>
RAJA_INLINE void forall_impl(cuda_occ_exec<Async>,
                             Iterable&& iter,
                             LoopBody&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);

  auto len = std::distance(begin, end);

  if (len > 0) {


    RAJA_FT_BEGIN;

    cudaStream_t stream = 0;


    // Ask occupancy calculator for optimal size
    int gridSize=-1, blockSize=-1;
    auto &launch_func = impl::forall_cuda_kernel_occ<decltype(begin), decltype(std::forward<LoopBody>(loop_body)), decltype(len)>;
    cudaOccupancyMaxPotentialBlockSize(
        &gridSize, &blockSize, launch_func, 0);


    // Compute number of logical blocks
    auto num_logical_blocks = impl::getGridDim(len, blockSize);

    // Restrict actual number of blocks, if it exceeds our iteration space
    gridSize = (num_logical_blocks < gridSize) ? num_logical_blocks : gridSize;


    // Launch
    impl::forall_cuda_kernel_occ<<<gridSize, blockSize, 0, stream>>>(
        RAJA::cuda::make_launch_body(
            gridSize, blockSize, 0, stream, std::forward<LoopBody>(loop_body)),
        std::move(begin),
        num_logical_blocks,
        len);
    RAJA::cuda::peekAtLastError();

    RAJA::cuda::launch(stream);
    if (!Async) RAJA::cuda::synchronize(stream);

    RAJA_FT_END;
  }
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables for within
// and device kernel.  Called from RAJA::nested::*
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody>
RAJA_INLINE RAJA_DEVICE void forall_impl(cuda_loop_exec,
                                         Iterable&& iter,
                                         LoopBody&& loop_body)
{
  // TODO: we need a portable std::begin, std::end, and std::distance
  auto begin = iter.begin();  // std::begin(iter);
  auto end = iter.end();      // std::end(iter);

  auto len = end - begin;  // std::distance(begin, end);

  for (decltype(len) i = 0; i < len; ++i) {
    loop_body(*(begin + i));
  }
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as CUDA kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_impl(ExecPolicy<seq_segit, cuda_exec<BlockSize, Async>>,
                             const TypedIndexSet<SegmentTypes...>& iset,
                             LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     detail::CallForall(),
                     cuda_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::cuda::synchronize();
}


template <typename LoopBody,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_impl(ExecPolicy<seq_segit, cuda_occ_exec<Async>>,
                             const TypedIndexSet<SegmentTypes...>& iset,
                             LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     detail::CallForall(),
                     cuda_occ_exec<true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::cuda::synchronize();
}



}  // closing brace for cuda namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
