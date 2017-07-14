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

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include <cstddef>

namespace RAJA
{

namespace cuda
{

struct pinned_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaHostAlloc(&ptr, nbytes, cudaHostAllocMapped));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFreeHost(ptr));
    return true;
  }

};

struct device_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }

};


struct device_zeroed_allocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaMalloc(&ptr, nbytes));
    cudaErrchk(cudaMemset(ptr, 0, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }

};

using device_mempool_type = basic_mempool::mempool<cuda::device_allocator>;
using device_zeroed_mempool_type = basic_mempool::mempool<cuda::device_zeroed_allocator>;
using pinned_mempool_type = basic_mempool::mempool<cuda::pinned_allocator>;

} // end namespace cuda


template <typename T, typename IndexType>
struct CudaReductionLocType {
  T val;
  IndexType idx;
};

template <typename T>
struct CudaReductionTallyType {
  T tally;
  unsigned int retiredBlocks;
};

template <typename T>
struct CudaReductionTallyTypeAtomic {
  T tally;
};

template <typename T, typename IndexType>
struct CudaReductionLocTallyType {
  CudaReductionLocType<T, IndexType> tally;
  unsigned int retiredBlocks;
};


void* getCudaReductionTallyBlockDeviceInternal(void* host_ptr);
void* getCudaReductionTallyBlockHostInternal(size_t size, size_t alignment = alignof(std::max_align_t));
void releaseCudaReductionTallyBlockHostInternal(void* host_ptr);


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

template <typename T>
void releaseCudaReductionTallyBlockHost(T* host_ptr)
{
  releaseCudaReductionTallyBlockHostInternal((void*)host_ptr);
}

void beforeCudaKernelLaunch(dim3 launchGridDim, dim3 launchBlockDim, cudaStream_t stream);
void afterCudaKernelLaunch(cudaStream_t stream);

void beforeCudaReadTallyBlockAsync(void* host_ptr);
void beforeCudaReadTallyBlockSync(void* host_ptr);

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
void freeCudaReductionMemBlockPool();

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
