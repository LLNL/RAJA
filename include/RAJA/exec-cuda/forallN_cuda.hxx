/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_cuda_HXX__
#define RAJA_forallN_cuda_HXX__

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

#include <cassert>

#include "RAJA/int_datatypes.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include <climits>

namespace RAJA
{

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-device constructor and device-only operator.
 */
template <typename BODY>
struct ForallN_BindFirstArg_Device {
  BODY const &body;
  size_t i;

  RAJA_INLINE
  RAJA_DEVICE
  constexpr ForallN_BindFirstArg_Device(BODY &b, size_t i0) : body(b), i(i0) {}

  template <typename... ARGS>
  RAJA_INLINE RAJA_DEVICE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};

/*!
 * \brief Struct that contains two CUDA dim3's that represent the number of
 * thread block and the number of blocks.
 *
 * This is passed to the execution policies to setup the kernel launch.
 */
struct CudaDim {
  dim3 num_threads;
  dim3 num_blocks;

  __host__ __device__ void print(void) const
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

template <typename POL>
struct CudaPolicy {
};

template <typename POL, typename IDX>
struct CudaIndexPair : public POL {
  template <typename IS>
  RAJA_INLINE constexpr CudaIndexPair(CudaDim &dims, IS const &is)
      : POL(dims, is)
  {
  }

  typedef IDX INDEX;
};

template <typename VIEWDIM, int threads_per_block>
struct CudaThreadBlock {
  int begin;
  int end;

  VIEWDIM view;

  CudaThreadBlock(CudaDim &dims, RangeSegment const &is)
      : begin(is.getBegin()), end(is.getEnd())
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = begin + view(blockIdx) * threads_per_block + view(threadIdx);
    if (idx >= end) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(CudaDim &dims)
  {
    int n = end - begin;
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

/*
 * These execution policies map a loop nest to the block and threads of a
 * given dimension with the number of THREADS per block specifies.
 */

template <int THREADS>
using cuda_threadblock_x_exec = CudaPolicy<CudaThreadBlock<Dim3x, THREADS>>;

template <int THREADS>
using cuda_threadblock_y_exec = CudaPolicy<CudaThreadBlock<Dim3y, THREADS>>;

template <int THREADS>
using cuda_threadblock_z_exec = CudaPolicy<CudaThreadBlock<Dim3z, THREADS>>;

template <typename VIEWDIM>
struct CudaThread {
  int begin;
  int end;

  VIEWDIM view;

  CudaThread(CudaDim &dims, RangeSegment const &is)
      : begin(is.getBegin()), end(is.getEnd())
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = begin + view(threadIdx);
    if (idx >= end) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(CudaDim &dims)
  {
    int n = end - begin;
    view(dims.num_threads) = n;
  }
};

/* These execution policies map the given loop nest to the threads in the
   specified dimensions (not blocks)
 */
using cuda_thread_x_exec = CudaPolicy<CudaThread<Dim3x>>;

using cuda_thread_y_exec = CudaPolicy<CudaThread<Dim3y>>;

using cuda_thread_z_exec = CudaPolicy<CudaThread<Dim3z>>;

template <typename VIEWDIM>
struct CudaBlock {
  int begin;
  int end;

  VIEWDIM view;

  CudaBlock(CudaDim &dims, RangeSegment const &is)
      : begin(is.getBegin()), end(is.getEnd())
  {
    setDims(dims);
  }

  __device__ inline int operator()(void)
  {
    int idx = begin + view(blockIdx);
    if (idx >= end) {
      idx = INT_MIN;
    }
    return idx;
  }

  void inline setDims(CudaDim &dims)
  {
    int n = end - begin;
    view(dims.num_blocks) = n;
  }
};

/* These execution policies map the given loop nest to the blocks in the
   specified dimensions (not threads)
 */
using cuda_block_x_exec = CudaPolicy<CudaBlock<Dim3x>>;

using cuda_block_y_exec = CudaPolicy<CudaBlock<Dim3y>>;

using cuda_block_z_exec = CudaPolicy<CudaBlock<Dim3z>>;

/*!
 * \brief  Function to check indices for out-of-bounds
 *
 */
template <typename BODY, typename... ARGS>
RAJA_INLINE __device__ void cudaCheckBounds(BODY &body, int i, ARGS... args)
{
  if (i > INT_MIN) {
    ForallN_BindFirstArg_Device<BODY> bound(body, i);
    cudaCheckBounds(bound, args...);
  }
}

template <typename BODY>
RAJA_INLINE __device__ void cudaCheckBounds(BODY &body, int i)
{
  if (i > INT_MIN) {
    body(i);
  }
}

/*!
 * \brief Launcher that uses execution policies to map blockIdx and threadIdx to map
 * to N-argument function
 */
template <typename BODY, typename... CARGS>
__global__ void cudaLauncherN(BODY loop_body, CARGS... cargs)
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Compute indices and then pass through the bounds-checking mechanism
  cudaCheckBounds(body, (cargs())...);
}

/*
 * The following is commented out.
 * It would be the preferred way of implementing the CUDA Executor
 * class, but there are some template deduction issues that need
 * to be worked out
 *
 */

template <int...>
struct integer_sequence {
};

template <int N, int... S>
struct gen_sequence : gen_sequence<N - 1, N - 1, S...> {
};

template <int... S>
struct gen_sequence<0, S...> {
  typedef integer_sequence<S...> type;
};

template <typename CuARG0,
          typename ISET0,
          typename CuARG1,
          typename ISET1,
          typename... CuARGS,
          typename... ISETS>
struct ForallN_Executor<ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>,
                        ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1>,
                        ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>...> {
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> iset0;
  ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> iset1;
  std::tuple<ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>...> isets;

  ForallN_Executor(
      ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const &iset0_,
      ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> const &iset1_,
      ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS> const &... isets_)
      : iset0(iset0_), iset1(iset1_), isets(isets_...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    unpackIndexSets(body, typename gen_sequence<sizeof...(CuARGS)>::type());
  }

  template <typename BODY, int... N>
  RAJA_INLINE void unpackIndexSets(BODY body, integer_sequence<N...>) const
  {
    CudaDim dims;

    callLauncher(dims,
                 body,
                 CuARG0(dims, iset0),
                 CuARG1(dims, iset1),
                 CuARGS(dims, std::get<N>(isets))...);
  }

  template <typename BODY, typename... CARGS>
  RAJA_INLINE void callLauncher(CudaDim const &dims,
                                BODY body,
                                CARGS const &... cargs) const
  {
    cudaLauncherN<<<dims.num_blocks,
                    dims.num_threads,
                    getCudaSharedmemAmount(dims.num_blocks, dims.num_threads)
                    >>>(body, cargs...);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};

template <typename CuARG0, typename ISET0>
struct ForallN_Executor<ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>> {
  ISET0 iset0;

  ForallN_Executor(ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const &iset0_)
      : iset0(iset0_)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    CudaDim dims;
    CuARG0 c0(dims, iset0);

    cudaLauncherN<<<dims.num_blocks,
                    dims.num_threads,
                    getCudaSharedmemAmount(dims.num_blocks, dims.num_threads)
                    >>>(body, c0);
    cudaErrchk(cudaPeekAtLastError());
    cudaErrchk(cudaDeviceSynchronize());
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
