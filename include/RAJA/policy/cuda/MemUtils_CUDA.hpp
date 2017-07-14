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

void synchronize();
void synchronize(cudaStream_t stream);

extern thread_local dim3 s_gridDim;
extern thread_local dim3 s_blockDim;
extern thread_local cudaStream_t s_stream;

RAJA_INLINE
dim3 currentGridDim()
{
  return s_gridDim;
}

RAJA_INLINE
dim3 currentBlockDim()
{
  return s_blockDim;
}

RAJA_INLINE
cudaStream_t currentStream()
{
  return s_stream;
}

template <typename T, typename IndexType>
struct LocType {
  T val;
  IndexType idx;
};

void beforeKernelLaunch(dim3 launchGridDim, dim3 launchBlockDim, cudaStream_t stream);
void afterKernelLaunch(bool Async);

void* getReductionMemBlockPoolInternal(size_t len, size_t size, size_t alignment = alignof(std::max_align_t));
void releaseReductionMemBlockPoolInternal(void* device_memblock);

template <typename T>
T* getReductionMemBlockPool()
{
  dim3 gridDim = currentGridDim();
  size_t len = gridDim.x * gridDim.y * gridDim.z;
  return (T*)getReductionMemBlockPoolInternal(len, sizeof(T), alignof(T));
}

template <typename T>
T* getReductionMemBlockPool(size_t len)
{
  return (T*)getReductionMemBlockPoolInternal(len, sizeof(T), alignof(T));
}

template <typename T>
void releaseReductionMemBlockPool(T *device_memblock)
{
  releaseReductionMemBlockPoolInternal((void*)device_memblock);
}

} // end namespace cuda

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
