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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_cuda_HPP
#define RAJA_forall_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <algorithm>

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/params/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

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
cuda_dim_t getGridDim(cuda_dim_member_t len, cuda_dim_t blockDim)
{
  cuda_dim_member_t block_size = blockDim.x * blockDim.y * blockDim.z;

  cuda_dim_member_t gridSize = (len + block_size - 1) / block_size;

  return {gridSize, 1, 1};
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 1D grid of 1D blocks
 *
 ******************************************************************************
 */
template <size_t BlockSize>
__device__ __forceinline__ unsigned int getGlobalIdx_1D_1D()
{
  unsigned int blockId = blockIdx.x;
  unsigned int threadId = blockId * BlockSize + threadIdx.x;
  return threadId;
}
template <size_t BlockSize>
__device__ __forceinline__ unsigned int getGlobalNumThreads_1D_1D()
{
  unsigned int numThreads = BlockSize * gridDim.x;
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
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
__launch_bounds__(BlockSize, BlocksPerSM) __global__
    void forall_cuda_kernel(LOOP_BODY loop_body,
                            const Iterator idx,
                            IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D<BlockSize>());
  if (ii < length) {
    body(idx[ii]);
  }
}

template <typename EXEC_POL,
          size_t BlockSize,
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam>
__launch_bounds__(BlockSize, 1) __global__
    void forallp_cuda_kernel(
                            LOOP_BODY loop_body,
                            const Iterator idx,
                            IndexType length,
                            ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D<BlockSize>());
  if ( ii < length )
  {
    RAJA::expt::invoke_body( f_params, body, idx[ii] );
  }
  RAJA::expt::ParamMultiplexer::combine<EXEC_POL>(f_params);
}

}  // namespace impl

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t BlockSize, size_t BlocksPerSM, bool Async, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Cuda>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Cuda cuda_res,
            cuda_exec_explicit<BlockSize, BlocksPerSM, Async>,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  auto func = impl::forall_cuda_kernel<BlockSize, BlocksPerSM, Iterator, LOOP_BODY, IndexType>;

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
    cuda_dim_t blockSize{BlockSize, 1, 1};
    cuda_dim_t gridSize = impl::getGridDim(static_cast<cuda_dim_member_t>(len), blockSize);

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
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          gridSize, blockSize, shmem, cuda_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::cuda::launch((const void*)func, gridSize, blockSize, args, shmem, cuda_res, Async);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(cuda_res);
}

template <typename Iterable, typename LoopBody, size_t BlockSize, size_t BlocksPerSM, bool Async, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Cuda>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate< RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>> >
forall_impl(resources::Cuda cuda_res,
            cuda_exec_explicit<BlockSize, BlocksPerSM, Async>,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = RAJA::cuda_exec_explicit<BlockSize, BlocksPerSM, Async>;

  auto func = impl::forallp_cuda_kernel< EXEC_POL, BlockSize, BlocksPerSM, Iterator, LOOP_BODY, IndexType, camp::decay<ForallParam> >;

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
    cuda_dim_t blockSize{BlockSize, 1, 1};
    cuda_dim_t gridSize = impl::getGridDim(static_cast<cuda_dim_member_t>(len), blockSize);

    RAJA_FT_BEGIN;

    RAJA::cuda::detail::cudaInfo launch_info;
    launch_info.gridDim = gridSize;
    launch_info.blockDim = blockSize;
    launch_info.res = cuda_res;

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //  printf("gridsize = (%d,%d), blocksize = %d\n",
    //         (int)gridSize.x,
    //         (int)gridSize.y,
    //         (int)blockSize.x);

    {
      RAJA::expt::ParamMultiplexer::init<EXEC_POL>(f_params, launch_info);
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          gridSize, blockSize, shmem, cuda_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len, (void*)&f_params};
      RAJA::cuda::launch((const void*)func, gridSize, blockSize, args, shmem, cuda_res, Async);

      RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(cuda_res);
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
          size_t BlocksPerSM,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Cuda>
forall_impl(resources::Cuda r,
            ExecPolicy<seq_segit, cuda_exec_explicit<BlockSize, BlocksPerSM, Async>>,
            const TypedIndexSet<SegmentTypes...>& iset,
            LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     cuda_exec_explicit<BlockSize, BlocksPerSM, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::cuda::synchronize(r);
  return resources::EventProxy<resources::Cuda>(r);
}

}  // namespace cuda

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
