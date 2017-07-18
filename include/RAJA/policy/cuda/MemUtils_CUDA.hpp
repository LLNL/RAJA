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

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/mutex.hpp"
#endif

#include <cstddef>
#include <cstdio>
#include <cassert>
#include <unordered_map>
#include <type_traits>

namespace RAJA
{

namespace cuda
{

template <typename T, typename IndexType>
struct LocType {
  T val;
  IndexType idx;
};

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

struct cudaInfo {
  dim3         gridDim  = 0;
  dim3         blockDim = 0;
  cudaStream_t stream   = 0;
  bool         active   = false;
#if defined(RAJA_ENABLE_OPENMP)
  cudaInfo*    thread_states = nullptr;
  omp::mutex   lock;
#endif
};

template < typename T >
class SetterResetter {
public:
  SetterResetter(T& val, T new_val)
    : m_val(val), m_old_val(val)
  {
    m_val = new_val;
  }
  ~SetterResetter()
  {
    m_val = m_old_val;
  }
private:
  T& m_val;
  T  m_old_val;
};

extern cudaInfo g_status;

extern std::unordered_map<cudaStream_t, bool> g_stream_info_map;

RAJA_INLINE
void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP)
  omp::lock_guard<omp::mutex> lock(g_status.lock);
#endif
  bool synchronize = false;
  for (auto& val : g_stream_info_map) {
    if (!val.second) {
      synchronize = true;
      val.second = true;
    }
  }
  if (synchronize) {
    cudaErrchk(cudaDeviceSynchronize());
  }
}

RAJA_INLINE
void synchronize(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP)
  omp::lock_guard<omp::mutex> lock(g_status.lock);
#endif
  auto iter = g_stream_info_map.find(stream);
  if (iter != g_stream_info_map.end() ) {
    if (!iter->second) {
      iter->second = true;
      cudaErrchk(cudaStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

RAJA_INLINE
void launch(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP)
  omp::lock_guard<omp::mutex> lock(g_status.lock);
#endif
  auto iter = g_stream_info_map.find(stream);
  if (iter != g_stream_info_map.end()) {
    iter->second = false;
  } else {
    g_stream_info_map.emplace(stream, false);
  }
}

// unthread-safe calls

RAJA_INLINE
bool currentlyActive()
{
#if defined(RAJA_ENABLE_OPENMP)
  {
    omp::lock_guard<omp::mutex> lock(g_status.lock);
    if (!g_status.thread_states) {
      g_status.thread_states = new cudaInfo[omp_get_max_threads()];
    }
  }
  return g_status.thread_states[omp_get_thread_num()].active;
#else
  return g_status.active;
#endif
}

RAJA_INLINE
dim3 currentGridDim()
{
#if defined(RAJA_ENABLE_OPENMP)
  return g_status.thread_states[omp_get_thread_num()].gridDim;
#else
  return g_status.gridDim;
#endif
}

RAJA_INLINE
dim3 currentBlockDim()
{
#if defined(RAJA_ENABLE_OPENMP)
  return g_status.thread_states[omp_get_thread_num()].blockDim;
#else
  return g_status.blockDim;
#endif
}

RAJA_INLINE
cudaStream_t currentStream()
{
#if defined(RAJA_ENABLE_OPENMP)
  return g_status.thread_states[omp_get_thread_num()].stream;
#else
  return g_status.stream;
#endif
}

template < typename LOOP_BODY >
RAJA_INLINE
typename std::remove_reference<LOOP_BODY>::type createLaunchBody(
  dim3 gridDim, dim3 blockDim, size_t dynamic_smem, cudaStream_t stream,
  LOOP_BODY&& loop_body)
{
#if defined(RAJA_ENABLE_OPENMP)
  {
    omp::lock_guard<omp::mutex> lock(g_status.lock);
    if (!g_status.thread_states) {
      g_status.thread_states = new cudaInfo[omp_get_max_threads()];
    }
  }
  int tid = omp_get_thread_num();
  cudaInfo& tl_status = g_status.thread_states[tid];
#else
  cudaInfo& tl_status = g_status;
#endif
  SetterResetter<bool> active_srer(tl_status.active, true);

  tl_status.stream   = stream;
  tl_status.gridDim  = gridDim;
  tl_status.blockDim = blockDim;

  return {loop_body};
}

} // end namespace cuda

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
