/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

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

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include <cassert>
#include <iostream>
#include <string>

namespace RAJA
{

namespace
{
/*!
 * \brief Number of currently active cuda reduction objects
 */
int s_cuda_reducer_active_count = 0;

/*!
 * \brief Number of cuda memblocks currently being used.
 */
int s_cuda_memblock_used_count = 0;

/*!
 * \brief Static array used to keep track of which unique ids
 * for CUDA reduction objects are used and which are not.
 */
bool s_cuda_reduction_id_used[RAJA_MAX_REDUCE_VARS];

/*!
 * \brief Static array used to keep track of which reduction
 * memblocks are in use.
 */
bool s_cuda_reduction_memblock_used[RAJA_MAX_REDUCE_VARS];

/*!
 * \brief Pointer to device memory block for RAJA-Cuda reductions.
 */
CudaReductionDummyBlockType* s_cuda_reduction_mem_block = 0;

/*!
 * \brief Pointer to the tally block on the device.
 *
 * The tally block is a number of contiguous slots in memory where the
 * results of cuda reduction variables are stored. This is done so all
 * results may be copied back to the host with one memcpy.
 */
CudaReductionDummyTallyType* s_cuda_reduction_tally_block_device = 0;

/*!
 * \brief Pointer to the tally block cache on the host.
 *
 * This cache allows multiple reads from the tally cache on the host to
 * incur only one memcpy from device memory. This cache also allows
 * multiple cuda reduction variables to be initialized without writing to
 * device memory. Changes to this cache are written back to the device
 * tally block in the next forall, before kernel launch so they are visible
 * on the device when needed.
 *
 * Note: This buffer must be allocated in pageable memory (not pinned).
 * CudaMemcpyAsync is always asynchronous with respect to managed memory.
 * However, while cudaMemcpyAsync is asynchronous to the host when used with
 * pinned or managed memory, it is synchronous to the host if the target
 * buffer is host pageable memory. Due to overheads associated with
 * synchronization of managed memory, using cudaMemcpyAsync with pageable
 * memory takes less time overall than using a synchronous routine. If
 * synchronizing managed memory incurs a smaller penalty inthe future, then
 * using other memory types could take less time.
 */
CudaReductionDummyTallyType* s_cuda_reduction_tally_block_host = 0;

//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of the tally block cache on the host.
//
/////////////////////////////////////////////////////////////////////////////
//

/*!
 * \brief Validity of host tally block cache.
 *
 * Valid means that all slots are up to date and can be read from the cache.
 * Invalid means that only dirty slots are up to date.
 */
bool s_tally_valid = true;
/*
 * \brief The number of slots that should be written back to the device
 *        tally block.
 */
int s_tally_dirty = 0;
/*
 * \brief Holds the dirty status of each slot.
 *
 * True indicates a slot written to by the host, but not copied back to
 * the device tally block.
 */
bool s_tally_block_dirty[RAJA_CUDA_REDUCE_TALLY_LENGTH] = {false};

//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of dynamic shared memory usage.
//
/////////////////////////////////////////////////////////////////////////////
//

/*!
 * \brief State of the host code, whether it is currently in a raja
 *        cuda forall function or not.
 */
int s_raja_cuda_forall_level = 0;
/*!
 * \brief The amount of shared memory currently earmarked for use in
 *        the current forall.
 */
int s_shared_memory_amount_total = 0;
/*!
 * \brief shared_memory_offsets holds the byte offsets into dynamic shared
 *        memory for each reduction variable.
 *
 * Note: -1 indicates a reduction variable that is not participating in
 * the current forall.
 */
int s_shared_memory_offsets[RAJA_MAX_REDUCE_VARS] = {-1};
/*!
 * \brief Holds the number of threads expected by each reduction variable.
 *
 * This is used to check the execution policy against the reduction policies
 * of participating reduction varaibles.
 *
 * Note: -1 indicates a reduction variable that is not participating in the
 * current forall and 0 represents a reduction variable whose execution does
 * not depend on the number of threads used by the execution policy.
 */
int s_cuda_reduction_num_threads[RAJA_MAX_REDUCE_VARS] = {-1};
}

/*
*******************************************************************************
*
* Return number of active cuda reducer objects.
*
*******************************************************************************
*/
int getCudaReducerActiveCount() { return s_cuda_reducer_active_count; }

/*
*******************************************************************************
*
* Return number of active cuda memblocks.
*
*******************************************************************************
*/
int getCudaMemblockUsedCount() { return s_cuda_memblock_used_count; }

/*
*******************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*******************************************************************************
*/
int getCudaReductionId()
{
  static int first_time_called = true;

  if (first_time_called) {
    s_cuda_reducer_active_count = 0;
    s_cuda_memblock_used_count = 0;

    for (int id = 0; id < RAJA_MAX_REDUCE_VARS; ++id) {
      s_cuda_reduction_id_used[id] = false;
    }

    first_time_called = false;
  }

  int id = 0;
  while (id < RAJA_MAX_REDUCE_VARS && s_cuda_reduction_id_used[id]) {
    id++;
  }

  if (id >= RAJA_MAX_REDUCE_VARS) {
    std::cerr << "\n Exceeded allowable RAJA CUDA reduction count, "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  s_cuda_reducer_active_count++;
  s_cuda_reduction_id_used[id] = true;

  return id;
}

/*
*******************************************************************************
*
* Release given reduction id and make inactive.
*
*******************************************************************************
*/
void releaseCudaReductionId(int id)
{
  if (id < RAJA_MAX_REDUCE_VARS) {
    s_cuda_reducer_active_count--;
    s_cuda_reduction_id_used[id] = false;
    if (s_cuda_reduction_memblock_used[id]) {
      s_cuda_memblock_used_count--;
      s_cuda_reduction_memblock_used[id] = false;
    }
  }
}

/*
*******************************************************************************
*
* Return pointer into RAJA-CUDA reduction device memory block
* for reducer object with given id. Allocate block if not already allocated.
*
*******************************************************************************
*/
void getCudaReductionMemBlock(int id, void** device_memblock)
{
  if (s_cuda_reduction_mem_block == 0) {
    cudaErrchk(
        cudaMalloc((void**)&s_cuda_reduction_mem_block,
                   sizeof(CudaReductionDummyBlockType) * RAJA_MAX_REDUCE_VARS));

    for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
      s_cuda_reduction_memblock_used[i] = false;
    }

    atexit(freeCudaReductionMemBlock);
  }

  s_cuda_memblock_used_count++;
  s_cuda_reduction_memblock_used[id] = true;

  *device_memblock = &(s_cuda_reduction_mem_block[id]);
}

/*
*******************************************************************************
*
* Free device memory blocks used in RAJA-Cuda reductions.
*
*******************************************************************************
*/
void freeCudaReductionMemBlock()
{
  if (s_cuda_reduction_mem_block != 0) {
    cudaErrchk(cudaFree(s_cuda_reduction_mem_block));
    s_cuda_reduction_mem_block = 0;
  }
}


/*
*******************************************************************************
*
* Return pointer into RAJA-CUDA reduction host tally block cache
* and device tally block for reducer object with given id.
* Allocate blocks if not already allocated.
*
*******************************************************************************
*/
void getCudaReductionTallyBlock(int id, void** host_tally, void** device_tally)
{
  if (s_cuda_reduction_tally_block_host == 0) {
    s_cuda_reduction_tally_block_host =
        new CudaReductionDummyTallyType[RAJA_CUDA_REDUCE_TALLY_LENGTH];

    cudaErrchk(cudaMalloc((void**)&s_cuda_reduction_tally_block_device,
                          sizeof(CudaReductionDummyTallyType)
                              * RAJA_CUDA_REDUCE_TALLY_LENGTH));

    s_tally_valid = true;
    s_tally_dirty = 0;
    for (int i = 0; i < RAJA_CUDA_REDUCE_TALLY_LENGTH; ++i) {
      s_tally_block_dirty[i] = false;
    }

    atexit(freeCudaReductionTallyBlock);
  }

  s_tally_dirty += 1;
  // set block dirty
  s_tally_block_dirty[id] = true;

  *host_tally = &(s_cuda_reduction_tally_block_host[id]);
  *device_tally = &(s_cuda_reduction_tally_block_device[id]);
}

/*
*******************************************************************************
*
* Write back dirty tally blocks to device tally blocks.
* Can be called before tally blocks have been allocated.
*
*******************************************************************************
*/
static void writeBackCudaReductionTallyBlock()
{
  if (s_tally_dirty > 0) {
    int first = 0;
    while (first < RAJA_CUDA_REDUCE_TALLY_LENGTH) {
      if (s_tally_block_dirty[first]) {
        int end = first + 1;
        while (end < RAJA_CUDA_REDUCE_TALLY_LENGTH
               && s_tally_block_dirty[end]) {
          end++;
        }
        int len = (end - first);
        cudaErrchk(cudaMemcpyAsync(&s_cuda_reduction_tally_block_device[first],
                                   &s_cuda_reduction_tally_block_host[first],
                                   sizeof(CudaReductionDummyTallyType) * len,
                                   cudaMemcpyHostToDevice));

        for (int i = first; i < end; ++i) {
          s_tally_block_dirty[i] = false;
        }
        first = end + 1;
      } else {
        first++;
      }
    }
    s_tally_dirty = 0;
  }
}

/*
*******************************************************************************
*
* Read tally block from device if invalid on host.
* Must be called after tally blocks have been allocated.
* The Async version is synchronous on the host if
* s_cuda_reduction_tally_block_host is allocated as pageable host memory
* and not if allocated as pinned host memory or managed memory.
*
*******************************************************************************
*/
static void readCudaReductionTallyBlockAsync()
{
  if (!s_tally_valid) {
    cudaErrchk(cudaMemcpyAsync(&s_cuda_reduction_tally_block_host[0],
                               &s_cuda_reduction_tally_block_device[0],
                               sizeof(CudaReductionDummyTallyType)
                                   * RAJA_CUDA_REDUCE_TALLY_LENGTH,
                               cudaMemcpyDeviceToHost));
    s_tally_valid = true;
  }
}
static void readCudaReductionTallyBlock()
{
  if (!s_tally_valid) {
    cudaErrchk(cudaMemcpy(&s_cuda_reduction_tally_block_host[0],
                          &s_cuda_reduction_tally_block_device[0],
                          sizeof(CudaReductionDummyTallyType)
                              * RAJA_CUDA_REDUCE_TALLY_LENGTH,
                          cudaMemcpyDeviceToHost));
    s_tally_valid = true;
  }
}

/*
*******************************************************************************
*
* Must be called before each RAJA cuda kernel, and before the copy of the
* loop body to setup state of the dynamic shared memory variables.
* Ensures that all updates to the tally block are visible on the device by
* writing back dirty cache lines; this invalidates the tally cache on the host.
*
*******************************************************************************
*/
void beforeCudaKernelLaunch()
{
  s_raja_cuda_forall_level++;
  if (s_raja_cuda_forall_level == 1) {
    if (s_cuda_reducer_active_count > 0) {
      s_shared_memory_amount_total = 0;
      for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
        s_shared_memory_offsets[i] = -1;
      }
      for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
        s_cuda_reduction_num_threads[i] = -1;
      }

      s_tally_valid = false;
      writeBackCudaReductionTallyBlock();
    }
  }
}

/*
*******************************************************************************
*
* Must be called after each RAJA cuda kernel.
* This resets the state of the dynamic shared memory variables.
*
*******************************************************************************
*/
void afterCudaKernelLaunch() { s_raja_cuda_forall_level--; }

/*
*******************************************************************************
*
* Must be called before reading the tally block cache on the host.
* Ensures that the host tally block cache for cuda reduction variable id can
* be read.
* Writes any host changes to the tally block cache to the device before
* updating the host tally blocks with the values on the GPU.
* The Async version is only asynchronous with regards to managed memory and
* is synchronous to host code.
*
*******************************************************************************
*/
void beforeCudaReadTallyBlockAsync(int id)
{
  if (!s_tally_block_dirty[id]) {
    writeBackCudaReductionTallyBlock();
    readCudaReductionTallyBlockAsync();
  }
}
///
void beforeCudaReadTallyBlockSync(int id)
{
  if (!s_tally_block_dirty[id]) {
    writeBackCudaReductionTallyBlock();
    readCudaReductionTallyBlock();
  }
}

/*
*******************************************************************************
*
* Release tally block of reduction variable with id.
*
*******************************************************************************
*/
void releaseCudaReductionTallyBlock(int id)
{
  if (s_tally_block_dirty[id]) {
    s_tally_block_dirty[id] = false;
    s_tally_dirty -= 1;
  }
}

/*
*******************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*******************************************************************************
*/
void freeCudaReductionTallyBlock()
{
  if (s_cuda_reduction_tally_block_host != 0) {
    delete[] s_cuda_reduction_tally_block_host;
    cudaErrchk(cudaFree(s_cuda_reduction_tally_block_device));
    s_cuda_reduction_tally_block_host = 0;
  }
}

/*
*******************************************************************************
*
* Earmark num_threads * size bytes of dynamic shared memory and get the byte
* offset.
*
*******************************************************************************
*/
int getCudaSharedmemOffset(int id, dim3 reductionBlockDim, int size)
{
  assert(id < RAJA_MAX_REDUCE_VARS);

  if (s_raja_cuda_forall_level > 0) {
    if (s_shared_memory_offsets[id] < 0) {
      // in a forall and have not yet gotten shared memory

      s_shared_memory_offsets[id] = s_shared_memory_amount_total;

      int num_threads =
          reductionBlockDim.x * reductionBlockDim.y * reductionBlockDim.z;

      // ignore reduction variables that don't use dynamic shared memory
      s_cuda_reduction_num_threads[id] = (size > 0) ? num_threads : 0;

      s_shared_memory_amount_total += num_threads * size;
    }
    return s_shared_memory_offsets[id];
  } else {
    return -1;
  }
}

/*
*******************************************************************************
*
* Get size in bytes of dynamic shared memory.
* Check that the number of blocks launched is consistent with the max number of
* blocks reduction variables can handle.
* Check that execution policy num_threads is consistent with active reduction
* policy num_threads.
*
*******************************************************************************
*/
int getCudaSharedmemAmount(dim3 launchGridDim, dim3 launchBlockDim)
{
  if (s_cuda_reducer_active_count > 0) {
    int launch_num_blocks = launchGridDim.x * launchGridDim.y * launchGridDim.z;

    int launch_num_threads =
        launchBlockDim.x * launchBlockDim.y * launchBlockDim.z;

    for (int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
      int reducer_num_threads = s_cuda_reduction_num_threads[i];

      // check if reducer is active
      if (reducer_num_threads >= 0) {

        // check if reducer cares about number of blocks
        if (s_cuda_reduction_memblock_used[i]
            && launch_num_blocks > RAJA_CUDA_MAX_NUM_BLOCKS) {
          std::cerr << "\n Cuda execution error: "
                    << "Can't launch " << launch_num_blocks << " blocks, "
                    << "RAJA_CUDA_MAX_NUM_BLOCKS = " << RAJA_CUDA_MAX_NUM_BLOCKS
                    << ", "
                    << "FILE: " << __FILE__ << " line: " << __LINE__
                    << std::endl;
          exit(1);
        }

        // check if reducer cares about number of threads
        if (reducer_num_threads > 0
            && launch_num_threads > reducer_num_threads) {
          std::cerr << "\n Cuda execution, reduction policy mismatch: "
                    << "reduction policy with BLOCK_SIZE "
                    << reducer_num_threads
                    << " can't be used with execution policy with BLOCK_SIZE "
                    << launch_num_threads << ", "
                    << "FILE: " << __FILE__ << " line: " << __LINE__
                    << std::endl;
          exit(1);
        }
      }
    }
  }
  return s_shared_memory_amount_total;
}

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_CUDA)
