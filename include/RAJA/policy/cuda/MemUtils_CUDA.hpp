/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_MemUtils_CUDA_HPP
#define RAJA_MemUtils_CUDA_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/types.hpp"

#include <cstddef>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief Type used to simplify hold value and location in cuda Loc reductions.
 *
 * Must fit within the dummy type (checked in static assert in the
 * reduction classes).
 *
 ******************************************************************************
 */
template <typename T, typename IndexType>
struct CudaReductionLocType {
  T val;
  IndexType idx;
};

/*!
 ******************************************************************************
 *
 * \brief Type used to simplify hold tally value in cuda reductions.
 *
 * Must fit within the dummy tally type (checked in static assert in the
 * reduction classes).
 *
 * Note: Retired blocks is used to count the number of blocks that finished
 * and wrote their portion of the reduction to the memory block.
 *
 ******************************************************************************
 */
template <typename T>
struct CudaReductionTallyType {
  T tally;
  unsigned int retiredBlocks;
};

/*!
 ******************************************************************************
 *
 * \brief Type used to simplify hold tally value in cuda atomic reductions.
 *
 * Must fit within the dummy tally type (checked in static assert in the
 * reduction classes).
 *
 ******************************************************************************
 */
template <typename T>
struct CudaReductionTallyTypeAtomic {
  T tally;
};

/*!
 ******************************************************************************
 *
 * \brief Type used to simplify hold tally value in cuda Loc reductions.
 *
 * Must fit within the dummy tally type (checked in static assert in the
 * reduction classes).
 *
 * Note: Retired blocks is used to count the number of blocks that finished
 * and wrote their portion of the reduction to the memory block.
 *
 ******************************************************************************
 */
template <typename T, typename IndexType>
struct CudaReductionLocTallyType {
  CudaReductionLocType<T, IndexType> tally;
  unsigned int retiredBlocks;
};


void* getCudaReductionTallyBlockDeviceInternal(void* host_ptr);
void* getCudaReductionTallyBlockHostInternal(size_t size, size_t alignment = alignof(std::max_align_t));
void releaseCudaReductionTallyBlockHostInternal(void* host_ptr);

/*!
 ******************************************************************************
 *
 * \brief Get tally block for reducer object with given id.
 *
 * \param[out] host_tally pointer to host tally cache slot.
 * \param[out] device_tally pointer to device tally slot.
 *
 * NOTE: Tally Block size will be:
 *
 *          sizeof(CudaReductionDummyTallyType) * RAJA_MAX_REDUCE_VARS
 *
 *       For each reducer object, we want a chunk of device memory that
 *       holds the reduced value and a small number of anciliary variables.
 *
 ******************************************************************************
 */

template <typename T>
T* getCudaReductionTallyBlockDevice(T* host_ptr)
{
  return (T*)getCudaReductionTallyBlockDeviceInternal((void*)host_ptr);
}

template <typename T>
T* getCudaReductionTallyBlockHost()
{
  return (T*)getCudaReductionTallyBlockHostInternal(sizeof(T), alignof(T));
}

/*!
 ******************************************************************************
 *
 * \brief Release tally block for reducer object with given id.
 *
 ******************************************************************************
 */
template <typename T>
void releaseCudaReductionTallyBlockHost(T* host_ptr)
{
  releaseCudaReductionTallyBlockHostInternal((void*)host_ptr);
}

/*!
 ******************************************************************************
 *
 * \brief Sets up state variales before the loop body is copied and the kernel
 *        is launched.
 *
 ******************************************************************************
 */
void beforeCudaKernelLaunch(dim3 launchGridDim, dim3 launchBlockDim, cudaStream_t stream);

/*!
 ******************************************************************************
 *
 * \brief Resets state variables after kernel launch.
 *
 ******************************************************************************
 */
void afterCudaKernelLaunch(cudaStream_t stream);

/*!
 ******************************************************************************
 *
 * \brief Updates host tally cache for read by reduction variable with id and
 * an asynchronous reduction policy.
 *
 ******************************************************************************
 */
void beforeCudaReadTallyBlockAsync(void* host_ptr);

/*!
 ******************************************************************************
 *
 * \brief Updates host tally cache for read by reduction variable with id and
 * a synchronous reduction policy.
 *
 ******************************************************************************
 */
void beforeCudaReadTallyBlockSync(void* host_ptr);

/*!
 ******************************************************************************
 *
 * \brief Updates host tally cache for read by reduction variable with id and
 * templated on Async from the reduction policy.
 *
 ******************************************************************************
 */
template <bool Async, typename T>
void beforeCudaReadTallyBlock(T* host_ptr)
{
  if (Async) {
    beforeCudaReadTallyBlockAsync((void*)host_ptr);
  } else {
    beforeCudaReadTallyBlockSync((void*)host_ptr);
  }
}


void* getCudaReductionMemBlockPoolInternal(size_t size, size_t alignment = alignof(std::max_align_t));
void releaseCudaReductionMemBlockPoolInternal(void* device_memblock);

/*!
 ******************************************************************************
 *
 * \brief  Get device memory block for RAJA-CUDA reduction variable  with
 *         given id.
 *
 *         Allocates data block if it isn't allocated already.
 *
 * \param[out] device_memblock Pointer to device memory block.
 *
 * NOTE: Total Block size will be:
 *
 *          sizeof(CudaReductionDummyDataType) *
 *            RAJA_MAX_REDUCE_VARS * RAJA_CUDA_REDUCE_BLOCK_LENGTH
 *
 *       For each reducer object, we want a chunk of device memory that
 *       holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction
 *       value for each thread block.
 *
 ******************************************************************************
 */
template <typename T>
T* getCudaReductionMemBlockPool()
{
  return (T*)getCudaReductionMemBlockPoolInternal(sizeof(T), alignof(T));
}

template <typename T>
void releaseCudaReductionMemBlockPool(T *device_memblock)
{
  releaseCudaReductionMemBlockPoolInternal((void*)device_memblock);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
