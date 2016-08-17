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

#include "RAJA/config.hxx"

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"

#include <iostream>
#include <string>
#include <cassert>

namespace RAJA
{

//
// Static array used to keep track of which unique ids
// for CUDA reduction objects are used and which are not.
//
static bool s_cuda_reduction_id_used[RAJA_MAX_REDUCE_VARS];

//
// Pointer to hold shared managed memory block for RAJA-Cuda reductions.
//
static CudaReductionDummyBlockType* s_cuda_reduction_mem_block = 0;

//
// Tally cache on the CPU
//
static CudaReductionDummyTallyType* s_cuda_reduction_tally_block_host = 0;

//
// Tally blocks on the device
//
static CudaReductionDummyTallyType* s_cuda_reduction_tally_block_device = 0;

static bool s_tally_valid = true;
static int s_tally_dirty = 0;
static bool s_tally_block_dirty[RAJA_CUDA_REDUCE_TALLY_LENGTH] = {false};

static bool s_in_raja_forall = false;
static int s_shared_memory_amount_total = 0;
static int s_shared_memory_offsets[RAJA_MAX_REDUCE_VARS] = {-1};
/*
*************************************************************************
*
* Return next available valid reduction id, or complain and exit if
* no valid id is available.
*
*************************************************************************
*/
int getCudaReductionId()
{
  static int first_time_called = true;

  if (first_time_called) {
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

  s_cuda_reduction_id_used[id] = true;

  return id;
}

/*
*************************************************************************
*
* Release given redution id and make inactive.
*
*************************************************************************
*/
void releaseCudaReductionId(int id)
{
  if (id < RAJA_MAX_REDUCE_VARS) {
    s_cuda_reduction_id_used[id] = false;
    s_tally_block_dirty[id] = false;
  }
}

/*
*************************************************************************
*
* Return pointer into shared RAJA-CUDA managed reduction memory block
* for reducer object with given id. Allocate block if not already allocated.
*
*************************************************************************
*/
void getCudaReductionMemBlock(int id, void** device_memblock)
{
  //
  // For each reducer object, we want a chunk of managed memory that
  // holds RAJA_CUDA_REDUCE_BLOCK_LENGTH slots for the reduction
  // value for each thread, a single slot for the global reduced value
  // across grid blocks, and a single slot for the max grid size.
  //

  if (s_cuda_reduction_mem_block == 0) {
    cudaErrchk(cudaMalloc((void**)&s_cuda_reduction_mem_block,
                          sizeof(CudaReductionDummyBlockType) * RAJA_MAX_REDUCE_VARS));

    atexit(freeCudaReductionMemBlock);
  }

  *device_memblock = &(s_cuda_reduction_mem_block[id]);
}

/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*************************************************************************
*/
void freeCudaReductionMemBlock()
{
  if (s_cuda_reduction_mem_block != 0) {
    cudaErrchk(cudaFree(s_cuda_reduction_mem_block));
    s_cuda_reduction_mem_block = 0;
  }
}



/*
*************************************************************************
*
* Return pointer into shared RAJA-CUDA reduction tally block
* for reducer object with given id. Return pointer to device tally block
* in device_tally.
* Allocate blocks if not already allocated.
*
*************************************************************************
*/
void getCudaReductionTallyBlock(int id, void** host_tally, void** device_tally)
{
  if (s_cuda_reduction_tally_block_host == 0) {
    s_cuda_reduction_tally_block_host = new CudaReductionDummyTallyType[RAJA_CUDA_REDUCE_TALLY_LENGTH];

    cudaErrchk(cudaMalloc((void**)&s_cuda_reduction_tally_block_device,
                          sizeof(CudaReductionDummyTallyType) * RAJA_CUDA_REDUCE_TALLY_LENGTH));

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

  *host_tally   = &(s_cuda_reduction_tally_block_host[id]);
  *device_tally = &(s_cuda_reduction_tally_block_device[id]);
}

/*
*************************************************************************
*
* Write back dirty tally blocks to device tally blocks.
* Can be called before tally blocks have been allocated.
*
*************************************************************************
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
        cudaErrchk(cudaMemcpyAsync( &s_cuda_reduction_tally_block_device[first],
                                    &s_cuda_reduction_tally_block_host[first],
                                    sizeof(CudaReductionDummyTallyType) * (end - first),
                                    cudaMemcpyHostToDevice, 0 ));
        
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
*************************************************************************
*
* Read tally block from device if invalid on host.
* Must be called after tally blocks have been allocated.
* This is synchronous if s_cuda_reduction_tally_block_host is allocated
* in pageable memory and not in pinned memory or managed.
*
*************************************************************************
*/
static void readCudaReductionTallyBlockAsync()
{
  if (!s_tally_valid) {
    cudaErrchk(cudaMemcpyAsync( &s_cuda_reduction_tally_block_host[0],
                                &s_cuda_reduction_tally_block_device[0],
                                sizeof(CudaReductionDummyTallyType) * RAJA_CUDA_REDUCE_TALLY_LENGTH,
                                cudaMemcpyDeviceToHost, 0 ));
    s_tally_valid = true;
  }
}

static void readCudaReductionTallyBlock()
{
  if (!s_tally_valid) {
    cudaErrchk(cudaMemcpy(  &s_cuda_reduction_tally_block_host[0],
                            &s_cuda_reduction_tally_block_device[0],
                            sizeof(CudaReductionDummyTallyType) * RAJA_CUDA_REDUCE_TALLY_LENGTH,
                            cudaMemcpyDeviceToHost));
    s_tally_valid = true;
  }
}

/*
*************************************************************************
*
* Must be called before each RAJA cuda kernel.
* Ensures all updates to teh tally block are visible on the gpu.
* Invalidates the tally on the CPU.
*
*************************************************************************
*/
void beforeCudaKernelLaunch()
{
  s_in_raja_forall = true;
  s_shared_memory_amount_total = 0;
  for(int i = 0; i < RAJA_MAX_REDUCE_VARS; ++i) {
    s_shared_memory_offsets[i] = -1;
  }

  s_tally_valid = false;
  writeBackCudaReductionTallyBlock();
}

void afterCudaKernelLaunch()
{
  s_in_raja_forall = false;
  s_shared_memory_amount_total = 0;
}

/*
*************************************************************************
*
* Must be called before reading a tally block on the CPU.
* Writes any CPU changes to the tally block back before updating the 
* CPU tally blocks with the values on the GPU.
*
*************************************************************************
*/
void beforeCudaReadTallyBlockAsync()
{
  writeBackCudaReductionTallyBlock();
  readCudaReductionTallyBlockAsync();
}

void beforeCudaReadTallyBlock()
{
  writeBackCudaReductionTallyBlock();
  readCudaReductionTallyBlock();
}


/*
*************************************************************************
*
* Release given redution tally block.
*
*************************************************************************
*/
void releaseCudaReductionTallyBlock(int id)
{
  if (s_tally_block_dirty[id]) {
    s_tally_block_dirty[id] = false;
    s_tally_dirty -= 1;
  }
}

/*
*************************************************************************
*
* Free managed memory blocks used in RAJA-Cuda reductions.
*
*************************************************************************
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
*************************************************************************
*
* Get offset into shared memory 
*
*************************************************************************
*/
int getCudaSharedmemOffset(int id, int amount)
{
  assert(id < RAJA_MAX_REDUCE_VARS);

  if (s_in_raja_forall) {
    if (s_shared_memory_offsets[id] < 0) {
      // in a forall and have not yet gotten shared memory

      s_shared_memory_offsets[id] = s_shared_memory_amount_total;

      s_shared_memory_amount_total += amount;
    }
    return s_shared_memory_offsets[id];
  } else {
    return -1;
  }
}

int getCudaSharedmemAmount()
{
  return s_shared_memory_amount_total;
}

}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_CUDA)
