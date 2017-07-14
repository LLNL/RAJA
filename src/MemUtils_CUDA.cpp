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

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/util/align.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"



#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif


#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <mutex>
#include <iostream>
#include <string>
#include <unordered_map>

namespace RAJA
{

namespace cuda
{

class TallyCache
{
public:

  static const size_t default_cache_size = 4*1024;
  static const size_t default_allocation_alignment  = 4*1024;
  static const size_t default_alignment  = alignof(std::max_align_t);

  explicit TallyCache(cudaStream_t stream, size_t allocation_size = default_cache_size, size_t allocation_alignment = default_allocation_alignment)
  {
    m_stream = stream;
    m_host_begin = (char*)RAJA::allocate_aligned(allocation_alignment, allocation_size*sizeof(char));
    m_host_end = m_host_begin+allocation_size;
    m_free_list = new cache_node{nullptr, m_host_begin, m_host_end, false};
    m_allocation_alignment = allocation_alignment;
  }

  bool active()
  {
    bool is_active = m_used_list != nullptr;
    if (!is_active && m_next) {
      is_active = m_next->active();
    }
    return is_active;
  }

  // not thread safe, assume called from 1 thread
  char* get_host_ptr(size_t size, size_t alignment = default_alignment)
  {
    char* host_ptr = nullptr;

    cache_node* free_node = m_free_list;

    if (free_node) {
      cache_node* prev_node = nullptr;

      while(free_node) {

        void* ptr = free_node->begin;
        size_t size_node = free_node->end - free_node->begin;

        if (RAJA::align(alignment, size, ptr, size_node)) {

          free_node = remove_free_node(free_node, prev_node, (char*)ptr, size);
          free_node = insert_used_node(free_node);

          m_count_dirty++;
          free_node->dirty = true;
          host_ptr = free_node->begin;
          break;
        }

      };

    }

    if (host_ptr == nullptr && m_used_list == nullptr) {
      std::cerr << "\n TallyCache coudn't handle request for " << size << " bytes at " << alignment << " alignment, "
                << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
      std::abort();
    }

    if (host_ptr == nullptr) {
      // ran out of room

      // allocate another cache with the same size
      if (!m_next) {
        m_next = new TallyCache(m_stream, m_host_end-m_host_begin, m_allocation_alignment);
      }

      // get a pointer from that cache
      host_ptr = m_next->get_host_ptr(size, alignment);
    }

    return host_ptr;
  }

  char* get_host_ptr(char* device_ptr)
  {
    char* host_ptr = nullptr;
    if (in_device_bounds(device_ptr)) {
      host_ptr = m_host_begin + (device_ptr - m_device_begin);
    } else if (m_next) {
      host_ptr = m_next->get_host_ptr(device_ptr);
    }
    return host_ptr;
  }

  char* get_device_ptr(char* host_ptr)
  {
    char* device_ptr = nullptr;

    if (!m_device_begin) {
      cudaErrchk(cudaMalloc(&m_device_begin, (m_host_end-m_host_begin)*sizeof(char)));
    }

    if (in_host_bounds(host_ptr)) {
      device_ptr = m_device_begin + (host_ptr - m_host_begin);
    } else if (m_next) {
      device_ptr = m_next->get_device_ptr(host_ptr);
    }
    return device_ptr;
  }

  void release_host_ptr(char* host_ptr)
  {
    if (in_host_bounds(host_ptr)) {

      cache_node* prev_node = nullptr;
      cache_node* node = m_used_list;
      while(node) {

        if (node->begin == host_ptr) break;

        prev_node = node;
        node = node->next;
      }

      if (node) {
        node = remove_used_node(node, prev_node);
        node = insert_free_node(node);
      } else {
        std::cerr << "\n TallyCache coudn't find node correspending to " << host_ptr << ", "
                  << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
        std::abort();
      }

    } else if (m_next) {
      m_next->release_host_ptr(host_ptr);
    }
  }

  void write_back_dirty()
  {
    if (m_next) {
      m_next->write_back_dirty();
    }

    if (m_count_dirty > 0) {

      cache_node* node = m_used_list;
      while (node) {
        if (node->dirty) {
          // found a run of dirty nodes
          char* host_end = node->end;
          node->dirty = false;

          while (node->next) {
            // advance to last dirty node in the run
            // this may include unused space between nodes
            if (node->next->dirty) {
              node = node->next;
              node->dirty = false;
            } else {
              break;
            }
          }

          char* host_begin = node->begin;

          size_t len = host_end - host_begin;
          char* device_begin = get_device_ptr(host_begin);

          cudaErrchk(cudaMemcpyAsync(device_begin, host_begin, 
                                     len*sizeof(char),
                                     cudaMemcpyHostToDevice, m_stream));
        }

        node = node->next;
      }

      m_count_dirty = 0;
    }
    
    m_valid = false;

  }

  void ensure_host_readable(char* host_ptr, bool async)
  {

    if (in_host_bounds(host_ptr)) {

      if (!m_valid) {

        cache_node* node = m_used_list;
        while (node) {
          if (!node->dirty) {
            // found a run of clean nodes
            char* host_end = node->end;

            while (node->next) {
              // advance to last clean node in the run
              // this may include unused space between nodes
              if (!node->next->dirty) {
                node = node->next;
              } else {
                break;
              }
            }

            char* host_begin = node->begin;

            size_t len = host_end - host_begin;
            char* device_begin = get_device_ptr(host_begin);

            cudaErrchk(cudaMemcpyAsync(host_begin, device_begin,
                                       len*sizeof(char),
                                       cudaMemcpyDeviceToHost,
                                       m_stream));
            if (!async) {
              cudaErrchk(cudaStreamSynchronize(m_stream));
            }

          }

          node = node->next;
        }

        m_valid = true;
      }

    }
    else if (m_next) {
      m_next->ensure_host_readable(host_ptr, async);
    }
  }

  ~TallyCache()
  {
    if (m_next) delete m_next;
    if (m_host_begin) free_aligned(m_host_begin);
    if (m_device_begin) cudaErrchk(cudaFree(m_device_begin));
  }

private:

  struct cache_node
  {
    cache_node* next;
    char* begin;
    char* end;
    bool  dirty;
  };

  cudaStream_t m_stream = 0;
  size_t m_allocation_alignment = default_allocation_alignment;

  TallyCache* m_next = nullptr;

  // stored in order
  cache_node* m_free_list  = nullptr;
  // stored in reverse order
  cache_node* m_used_list  = nullptr;
  // extra nodes to avoid unnecessary calls to new/delete
  cache_node* m_extra_list = nullptr;

  char* m_host_begin   = nullptr;
  char* m_host_end     = nullptr;
  char* m_device_begin = nullptr;
  char* m_device_end   = nullptr;

  unsigned m_count_dirty = 0u;
  bool m_valid = true;

  bool in_host_bounds(char* host_ptr)
  {
    return m_host_begin <= host_ptr && host_ptr < m_host_end;
  }

  bool in_device_bounds(char* device_ptr)
  {
    return m_device_begin <= device_ptr && device_ptr < m_device_end;
  }

  cache_node* pop_extra_node()
  {
    cache_node* extra = m_extra_list;
    if (extra) {
      m_extra_list = extra->next;
      extra->next = nullptr;
    } else {
      extra = new cache_node{nullptr, nullptr, nullptr, false};
    }
    return extra;
  }

  void push_extra_node(cache_node* extra)
  {
    extra->next  = m_extra_list;
    extra->begin = nullptr;
    extra->end   = nullptr;
    extra->dirty = false;

    m_extra_list = extra;
  }

  // returns a node for ptr, size by splitting free_node
  cache_node* remove_free_node(cache_node* free_node, cache_node* prev_node, char* ptr, size_t size)
  {
    if (ptr+size < free_node->end) {
      // make new node between free and next
      cache_node* next_node = pop_extra_node();

      next_node->next  = free_node->next;
      next_node->begin = ptr+size;
      next_node->end   = free_node->end;

      free_node->next  = next_node;
      free_node->end   = ptr+size;
    }

    if (free_node->begin != ptr) {
      // starting at an offset into free node
      if (prev_node) {
        // make new node between prev, free
        prev_node->next = pop_extra_node();
        prev_node = prev_node->next;
      } else {
        // make new node at head of free list
        m_free_list = pop_extra_node();
        prev_node = m_free_list;
      }

      prev_node->next  = free_node;
      prev_node->begin = free_node->begin;
      prev_node->end   = ptr;

      free_node->begin = ptr;
    }

    if (prev_node) {
      prev_node->next = free_node->next;
    } else {
      m_free_list = free_node->next;
    }

    free_node->next = nullptr;

    return free_node;
  }

  cache_node* insert_used_node(cache_node* used_node)
  {
    // traverse used list until the next node is less than used node
    cache_node* prev_node = nullptr;
    cache_node* next_node = m_used_list;
    while (next_node) {

      if (next_node->begin < used_node->begin) break;

      prev_node = next_node;
      next_node = next_node->next;
    }

    if (prev_node) {
      used_node->next = prev_node->next;
      prev_node->next = used_node;
    } else {
      used_node->next = m_used_list;
      m_used_list = used_node;
    }
    return used_node;
  }


  // returns used_node
  cache_node* remove_used_node(cache_node* used_node, cache_node* prev_node)
  {

    if (prev_node) {
      prev_node->next = used_node->next;
    } else {
      m_used_list = used_node->next;
    }

    used_node->next = nullptr;

    return used_node;
  }

  // returns the node contatining free_node's memory
  cache_node* insert_free_node(cache_node* free_node)
  {
    // traverse free list until the next node is greater than free node
    cache_node* prev_node = nullptr;
    cache_node* next_node = m_free_list;
    while (next_node) {

      if (next_node->begin >= free_node->begin) break;

      prev_node = next_node;
      next_node = next_node->next;
    }

    if (prev_node) {
      if (prev_node->end == free_node->begin) {
        // merge free node into prev node

        prev_node->end = free_node->end;

        push_extra_node(free_node);

        free_node = prev_node;

      } else {
        // insert free node after prev node
        prev_node->next = free_node;
        free_node->next = next_node;
      }

    } else {
      free_node->next = m_free_list;
      m_free_list = free_node;
    }

    if (next_node) {
      if (free_node->end == next_node->begin) {
        // merge next node into free node

        free_node->next = next_node->next;
        free_node->end  = next_node->end;

        push_extra_node(next_node);
      }
    }

    return free_node;
  }

};


device_mempool_type& s_cuda_reduction_mem_block_pool = device_mempool_type::getInstance();

/*!
 * \brief Pointer to the tally block on the device.
 *
 * The tally block is a number of contiguous slots in memory where the
 * results of cuda reduction variables are stored. This is done so all
 * results may be copied back to the host with one memcpy.
 */

TallyCache* s_tally_cache = nullptr;

//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

/*!
 * \brief State of the host code, whether it is currently in a raja
 *        cuda forall function or not.
 */
thread_local int s_forall_level = 0;

thread_local dim3 s_launchGridDim = 0;
thread_local dim3 s_launchBlockDim = 0;

thread_local cudaStream_t s_stream = 0;

std::unordered_map<cudaStream_t, bool> s_stream_info(std::unordered_map<cuda_stream, bool>::value_type(0, true));

void synchronize()
{
  bool synchronize = false;
  for (auto& val : s_stream_info) {
    if (!val.second) {
      synchronize = true;
      val.second = true;
    }
  }
  if (synchronize) {
    cudaErrchk(cudaDeviceSynchronize());
  }
}

void synchronize(cudaStream_t stream)
{
  auto iter = s_stream_info.find(stream);
  if (iter != s_stream_info.end() ) {
    if (!iter->second) {
      iter->second = true;
      cudaErrchk(cudaStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

void launch(cudaStream_t stream)
{
  auto iter = s_stream_info.find(stream);
  if (iter != s_stream_info.end()) {
    iter->second = false;
  } else {
    fprintf(stderr, "Cannot launch using unknown stream.\n");
    std::abort();
  }
}


void* getReductionMemBlockPoolInternal(size_t len, size_t size, size_t alignment)
{
  void* ptr = nullptr;

#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (MemUtils_CUDA)
  {
#endif
    // assume beforeCudaKernelLaunch was called in order to grab dimensions 
    if(s_forall_level > 0) {
      ptr = (void*)s_cuda_reduction_mem_block_pool.malloc<char>(len*size, alignment);
    }
#if defined(RAJA_ENABLE_OPENMP)
  }
#endif

  return ptr;
}

void releaseReductionMemBlockPoolInternal(void *device_memblock)
{
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (MemUtils_CUDA)
  {
#endif
    if(device_memblock) {
      s_cuda_reduction_mem_block_pool.free(device_memblock);
    }
#if defined(RAJA_ENABLE_OPENMP)
  }
#endif
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
void beforeKernelLaunch(dim3 launchGridDim, dim3 launchBlockDim, cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (MemUtils_CUDA)
  {  
#endif
    s_forall_level++;

    if (s_forall_level == 1 && s_tally_cache) {

      if (s_tally_cache->active()) {
        s_tally_cache->write_back_dirty();
      }
    }

    s_launchGridDim = launchGridDim;
    s_launchBlockDim = launchBlockDim;
    s_stream = stream;
#if defined(RAJA_ENABLE_OPENMP)
  }
#endif
}

/*
*******************************************************************************
*
* Must be called after each RAJA cuda kernel.
* This resets the state of the dynamic shared memory variables.
*
*******************************************************************************
*/
void afterKernelLaunch(bool Async)
{
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (MemUtils_CUDA)
  {  
#endif
    s_cuda_launch_gridDim = 0;
    s_cuda_launch_blockDim = 0;

    cuda::launch(s_stream);

    s_raja_cuda_forall_level--;
    
    cudaErrchk(cudaPeekAtLastError());
    if (!Async) {
      cuda::synchronize(s_stream);
    }
    s_stream = 0;
#if defined(RAJA_ENABLE_OPENMP)
  }
#endif
}


}  // closing brace for RAJA namespace

#endif  // if defined(RAJA_ENABLE_CUDA)
