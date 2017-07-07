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
 * \def RAJA_CUDA_REDUCE_VAR_MAXSIZE
 * Size in bytes used in CudaReductionDummyDataType for array allocation to
 * accommodate the template type used in reductions.
 *
 * Note: Includes the size of the index variable for Loc reductions.
 */
#define RAJA_CUDA_REDUCE_VAR_MAXSIZE 16

/*!
 * \brief Type used to keep track of the grid size on the device
 */
typedef unsigned int GridSizeType;

/*!
 ******************************************************************************
 *
 * \brief Type representing a single typed value for a cuda reduction.
 *
 * Enough space for a double value and an index value.
 *
 ******************************************************************************
 */
struct RAJA_ALIGNED_ATTR(RAJA_CUDA_REDUCE_VAR_MAXSIZE)
    CudaReductionDummyDataType {
  unsigned char data[RAJA_CUDA_REDUCE_VAR_MAXSIZE];
};

/*!
 ******************************************************************************
 *
 * \brief Type representing a memory block for a cuda reduction.
 *
 ******************************************************************************
 */

typedef CudaReductionDummyDataType CudaReductionDummyBlockType;

/*!
 ******************************************************************************
 *
 * \brief Type representing enough memory to hold a slot in the tally block.
 *
 ******************************************************************************
 */
struct CudaReductionDummyTallyType {
  CudaReductionDummyDataType dummy_val;
  GridSizeType dummy_retiredBlocks;
};

/*!
 ******************************************************************************
 *
 * \brief Type used to simplify typed memory block use in cuda Loc reductions.
 *
 * Must fit within the dummy block type (checked in static assert in the
 * reduction classes).
 *
 ******************************************************************************
 */

template <typename T>
struct CudaReductionLocBlockType {
  T value;
  Index_type index;
};

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
template <typename T>
struct CudaReductionLocType {
  T val;
  Index_type idx;
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
  GridSizeType retiredBlocks;
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
template <typename T>
struct CudaReductionLocTallyType {
  CudaReductionLocType<T> tally;
  GridSizeType retiredBlocks;
};

/*!
 ******************************************************************************
 *
 * \brief Set the Max Number of Blocks that RAJA will launch
 *
 * Modulates the memblock size that non-atomic reducers use 
 *
 * \return bool true for success, false for failure
 *
 ******************************************************************************
 */
void setCudaMaxBlocks(unsigned int blocks);

/*!
 ******************************************************************************
 *
 * \brief Set the Max Number of Reducers that RAJA will launch
 *
 * Modulates the memblock size that non-atomic reducers use 
 *
 * \return bool true for success, false for failure
 *
 ******************************************************************************
 */
void setCudaMaxReducers(unsigned int reducers);

/*!
 ******************************************************************************
 *
 * \brief Get the number of active cuda reducer objects.
 *
 * \return int number of active cuda reducer objects.
 *
 ******************************************************************************
 */
bool getCudaReducerActive();

/*!
 ******************************************************************************
 *
 * \brief Get a valid reduction id, or complain and exit if no valid id is
 *        available.
 *
 * \return int the next available valid reduction id.
 *
 ******************************************************************************
 */
int getCudaReductionId();

/*!
 ******************************************************************************
 *
 * \brief Release given reduction id and make inactive.
 *
 ******************************************************************************
 */
void releaseCudaReductionId(int id);

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
void* getCudaReductionTallyBlockDeviceInternal(void* host_ptr);
void* getCudaReductionTallyBlockHostInternal(size_t size, size_t alignment = alignof(std::max_align_t));

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
void releaseCudaReductionTallyBlockHostInternal(void* host_ptr);

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
void beforeCudaKernelLaunch(dim3 launchGridDim, dim3 launchBlockDim);

/*!
 ******************************************************************************
 *
 * \brief Resets state variables after kernel launch.
 *
 ******************************************************************************
 */
void afterCudaKernelLaunch();

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
void* getCudaReductionMemBlockInternal(int id, size_t size, size_t alignment = alignof(std::max_align_t));

template <typename T>
T* getCudaReductionMemBlock(int id)
{
  return (T*)getCudaReductionMemBlockInternal(id, sizeof(T), alignof(T));
}

/*!
 ******************************************************************************
 *
 * \brief  Free device memory blocks used in RAJA-Cuda reductions.
 *
 ******************************************************************************
 */
void freeCudaReductionMemBlock();

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
