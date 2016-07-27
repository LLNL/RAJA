/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cuda_HXX
#define RAJA_forall_cuda_HXX

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

#include "RAJA/int_datatypes.hxx"

#include "RAJA/fault_tolerance.hxx"

#include "RAJA/exec-cuda/raja_cudaerrchk.hxx"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//
/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <typename Iterator, typename LOOP_BODY>
__global__ void forall_cuda_kernel(LOOP_BODY loop_body,
                                   const Iterator idx,
                                   Index_type length)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall_Icount template for indiraction array.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename Iterator, typename LOOP_BODY>
__global__ void forall_Icount_cuda_kernel(LOOP_BODY loop_body,
                                          const Iterator idx,
                                          Index_type length,
                                          Index_type icount)
{
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii + icount, idx[ii]);
  }
}

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <size_t BLOCK_SIZE, bool Async, typename Iterable, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec<BLOCK_SIZE, Async>,
                        Iterable&& iter,
                        LOOP_BODY loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  Index_type len = std::distance(begin, end);

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(std::move(loop_body),
                                               std::move(begin),
                                               len);
  cudaErrchk(cudaPeekAtLastError());
  if (!Async) {
    cudaErrchk(cudaDeviceSynchronize());
  }

  RAJA_FT_END;
}

template <size_t BLOCK_SIZE, bool Async, typename Iterable, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec<BLOCK_SIZE, Async>,
                               Iterable&& iter,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  Index_type len = std::distance(begin, end);

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(std::move(loop_body),
                                                      std::move(begin),
                                                      len,
                                                      icount);
  cudaErrchk(cudaPeekAtLastError());
  if (!Async) {
    cudaErrchk(cudaDeviceSynchronize());
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as CUDA kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(IndexSet::ExecPolicy<seq_segit, cuda_exec<BLOCK_SIZE>>,
                        const IndexSet& iset,
                        LOOP_BODY loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
    executeRangeList_forall<cuda_exec_async<BLOCK_SIZE>>(seg_info, loop_body);

  }  // iterate over segments of index set

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 *         This method passes index count to segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(
    IndexSet::ExecPolicy<seq_segit, cuda_exec<BLOCK_SIZE>>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
    executeRangeList_forall_Icount<cuda_exec_async<BLOCK_SIZE>>(seg_info,
                                                                loop_body);

  }  // iterate over segments of index set

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());
}

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
