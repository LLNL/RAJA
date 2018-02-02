/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via ROCM kernel launch.
 *
 *          These methods should work on any platform that supports
 *          ROCM devices.
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

#ifndef RAJA_forall_rocm_HPP
#define RAJA_forall_rocm_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/pattern/forall.hpp"

#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"
#include "RAJA/policy/rocm/policy.hpp"

#include "RAJA/index/IndexSet.hpp"

#include <algorithm>


namespace RAJA
{

namespace policy
{
namespace rocm
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
dim3 getGridDim(size_t len, size_t blockDim) 
{
  size_t block_size = blockDim;

  size_t gridSize = (len + block_size - 1) / block_size;

  return gridSize;
}
RAJA_INLINE
dim3 getGridDim(size_t len, dim3 blockDim) 
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
__forceinline__ unsigned int getGlobalIdx_1D_1D() [[hc]]
{
  unsigned int blockId = blockIdx_x;
  unsigned int threadId = blockId * blockDim_x + threadIdx_x;
  return threadId;
}
__forceinline__ unsigned int getGlobalNumThreads_1D_1D() [[hc]]
{
  unsigned int numThreads = blockDim_x * gridDim_x;
  return numThreads;
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 3D grid of 3D blocks
 *
 ******************************************************************************
 */
__inline__ unsigned int getGlobalIdx_3D_3D() [[hc]]
{
  unsigned int blockId =
      blockIdx_x + blockIdx_y * gridDim_x + gridDim_x * gridDim_y * blockIdx_z;
  unsigned int threadId = blockId * (blockDim_x * blockDim_y * blockDim_z)
                          + (threadIdx_z * (blockDim_x * blockDim_y))
                          + (threadIdx_y * blockDim_x) + threadIdx_x;
  return threadId;
}
__inline__ unsigned int getGlobalNumThreads_3D_3D() [[hc]]
{
  unsigned int numThreads =
      blockDim_x * blockDim_y * blockDim_z * gridDim_x * gridDim_y * gridDim_z;
  return numThreads;
}

//
//////////////////////////////////////////////////////////////////////
//
// ROCM kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  ROCM kernel forall template for indirection array.
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
//__launch_bounds__(BlockSize, 1) 
    inline 
    void forall_rocm_kernel(LOOP_BODY loop_body,
                            const Iterator idx,
                            IndexType length) [[hc]]
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto &body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  if (ii < length) {
    body(idx[ii]);
  }
}




}  // end impl namespace

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for ROCM execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE void forall_impl(rocm_exec<BlockSize, Async>,
                             Iterable&& iter,
                             LoopBody&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);

  auto len = std::distance(begin, end);

  if (len > 0 && BlockSize > 0) {

    auto gridSize = impl::getGridDim(len, BlockSize);

    RAJA_FT_BEGIN;

    rocmStream_t stream = 0;
    dim3 block(BlockSize,1,1);
    dim3 grid(len,1,1);

//    hc::parallel_for_each(ext.tile_with_dynamic(block.x,block.y,block.z,shmem), [=](const hc::index<3> & idx) [[hc]] [[hc]]
    if ( grid.x && ( block.x * block.y * block.z ) ) {
//         LoopBody * rocm_device_buffer = (LoopBody *)
//                                 rocmDeviceAlloc(sizeof(LoopBody));


      // Copy functor to constant memory on the device
//      rocm_device_copy(loop_body,rocm_device_buffer,sizeof(LoopBody));

	auto ext = hc::extent<3>(grid.x,grid.y,grid.z);
        auto fut = hc::parallel_for_each(ext.tile(block.x,block.y,block.z), 
                                         [=](const hc::index<3> & idx) [[hc]]{ 
        impl::forall_rocm_kernel<BlockSize>(
           RAJA::rocm::make_launch_body(
             gridSize, BlockSize, 0, stream, std::forward<LoopBody>(loop_body)),
             std::move(begin),
             len);
      }).wait();
//      rocmDeviceFree(rocm_device_buffer);
    }


    RAJA::rocm::launch(stream);
    if (!Async) RAJA::rocm::synchronize(stream);

   RAJA_FT_END;
  }
}


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for ROCM execution over iterables for within
// and device kernel.  Called from RAJA::nested::*
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody>
RAJA_INLINE RAJA_DEVICE void forall_impl(rocm_loop_exec,
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
// segments as ROCM kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         ROCM execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_impl(ExecPolicy<seq_segit, rocm_exec<BlockSize, Async>>,
                             const TypedIndexSet<SegmentTypes...>& iset,
                             LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     detail::CallForall(),
                     rocm_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::rocm::synchronize();
}

}  // closing brace for rocm namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
