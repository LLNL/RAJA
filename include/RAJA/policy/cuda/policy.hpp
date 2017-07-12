/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA CUDA policy definitions.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_HPP
#define RAJA_policy_cuda_HPP

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

#include "RAJA/config.hpp"
#include "RAJA/policy/PolicyBase.hpp"

//
// Operations in the included files are parametrized using the following
// values for CUDA warp size and max block size.
//
const int WARP_SIZE = 32;
const int RAJA_CUDA_MAX_BLOCK_SIZE = 2048;

/*!
 * \def RAJA_CUDA_LAUNCH_PARAMS(gridSize, blockSize)
 * Macro that generates kernel launch parameters.
 */
#define RAJA_CUDA_LAUNCH_PARAMS(gridSize, blockSize) \
  gridSize, blockSize, getCudaSharedmemAmount(gridSize, blockSize)

namespace RAJA
{
namespace cuda
{
namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr Launch value = Async ? Launch::async : Launch::sync;
};
}

#if defined(RAJA_ENABLE_CLANG_CUDA)
using dim_t = uint3;
#else
using dim_t = dim3;
#endif

/*!
 * \brief Struct that contains two CUDA dim3's that represent the number of
 * thread block and the number of blocks.
 *
 * This is passed to the execution policies to setup the kernel launch.
 */
struct Dim {
  dim_t num_threads;
  dim_t num_blocks;

  RAJA_HOST_DEVICE void print(void) const
  {
    printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
           num_blocks.x,
           num_blocks.y,
           num_blocks.z,
           num_threads.x,
           num_threads.y,
           num_threads.z);
  }
};

template <typename POL, typename IDX>
struct IndexPair : public POL {
  template <typename IS>
  RAJA_INLINE constexpr IndexPair(Dim &dims, IS const &is) : POL(dims, is)
  {
  }

  using INDEX = IDX;
};

/** Provides a range from 0 to N_iter - 1
 *
 */
template <typename VIEWDIM, int threads_per_block>
struct ThreadBlock {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  ThreadBlock(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = 0 + view(blockIdx) * threads_per_block + view(threadIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims)
  {
    int n = distance;
    if (n < threads_per_block) {
      view(dims.num_threads) = n;
      view(dims.num_blocks) = 1;
    } else {
      view(dims.num_threads) = threads_per_block;

      int blocks = n / threads_per_block;
      if (n % threads_per_block) {
        ++blocks;
      }
      view(dims.num_blocks) = blocks;
    }
  }
};

template <typename VIEWDIM>
struct Thread {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  Thread(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = view(threadIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims) { view(dims.num_threads) = distance; }
};

template <typename VIEWDIM>
struct Block {
  int distance;

  VIEWDIM view;

  template <typename Iterable>
  Block(Dim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = view(blockIdx);
    if (idx >= distance) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(Dim &dims) { view(dims.num_blocks) = distance; }
};


///
/////////////////////////////////////////////////////////////////////
///
/// Generalizations of CUDA dim3 x, y and z used to describe
/// sizes and indices for threads and blocks.
///
/////////////////////////////////////////////////////////////////////
///
struct Dim3x {
  __host__ __device__ inline unsigned int &operator()(dim_t &dim)
  {
    return dim.x;
  }

  __host__ __device__ inline unsigned int operator()(dim_t const &dim)
  {
    return dim.x;
  }
};
///
struct Dim3y {
  __host__ __device__ inline unsigned int &operator()(dim_t &dim)
  {
    return dim.y;
  }

  __host__ __device__ inline unsigned int operator()(dim_t const &dim)
  {
    return dim.y;
  }
};
///
struct Dim3z {
  __host__ __device__ inline unsigned int &operator()(dim_t &dim)
  {
    return dim.z;
  }

  __host__ __device__ inline unsigned int operator()(dim_t const &dim)
  {
    return dim.z;
  }
};

} // end namespace cuda

} // end namespace RAJA

#include "RAJA/policy/cuda/impl/atomics.hpp"

namespace RAJA
{

//
/////////////////////////////////////////////////////////////////////
//
// Execution policies
//
/////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_exec :
    make_policy_pattern_launch_platform_t<Policy::cuda,
                                          Pattern::forall,
                                          cuda::detail::get_launch<Async>::value,
                                          Platform::cuda>{};

//
// NOTE: There is no Index set segment iteration policy for CUDA
//

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_reduce :
    make_policy_pattern_launch_t<Policy::cuda,
                                 Pattern::reduce,
                                 Async ? Launch::async : Launch::sync>{};

template <size_t BLOCK_SIZE>
struct cuda_reduce_async : cuda_reduce<BLOCK_SIZE, true>{};

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_reduce_atomic :
    make_policy_pattern_launch_t<Policy::cuda,
                                 Pattern::reduce,
                                 Async ? Launch::async : Launch::sync>{};

template <size_t BLOCK_SIZE>
struct cuda_reduce_atomic_async : cuda_reduce_atomic<BLOCK_SIZE, true>{};

template <typename... Args>
struct CudaPolicy : make_policy_pattern_launch_platform_t<Policy::cuda,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::cuda,
                                                         Args...>{};

/*
 * These execution policies map a loop nest to the block and threads of a
 * given dimension with the number of THREADS per block specifies.
 */

template <int THREADS>
struct cuda_threadblock_x_exec :
    CudaPolicy<cuda::ThreadBlock<cuda::Dim3x, THREADS>>{};

template <int THREADS>
struct cuda_threadblock_y_exec :
    CudaPolicy<cuda::ThreadBlock<cuda::Dim3y, THREADS>>{};

template <int THREADS>
struct cuda_threadblock_z_exec :
    CudaPolicy<cuda::ThreadBlock<cuda::Dim3z, THREADS>>{};

/* These execution policies map the given loop nest to the threads in the
   specified dimensions (not blocks)
 */
struct cuda_thread_x_exec : CudaPolicy<cuda::Thread<cuda::Dim3x>>{};

struct cuda_thread_y_exec : CudaPolicy<cuda::Thread<cuda::Dim3y>>{};

struct cuda_thread_z_exec : CudaPolicy<cuda::Thread<cuda::Dim3z>>{};

/* These execution policies map the given loop nest to the blocks in the
   specified dimensions (not threads)
 */
struct cuda_block_x_exec : CudaPolicy<cuda::Block<cuda::Dim3x>>{};

struct cuda_block_y_exec : CudaPolicy<cuda::Block<cuda::Dim3y>>{};

struct cuda_block_z_exec : CudaPolicy<cuda::Block<cuda::Dim3z>>{};

}  // closing brace for RAJA namespace

#endif
