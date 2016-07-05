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

namespace RAJA {

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
 * \brief CUDA kernal forall template for index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_cuda_kernel(LOOP_BODY loop_body,
                                   Index_type begin,
                                   Index_type len) {
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < len) {
    loop_body(begin + ii);
  }
}

/*!
 ******************************************************************************
 *
 * \brief CUDA kernal forall_Icount template for index range.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_Icount_cuda_kernel(LOOP_BODY loop_body,
                                          Index_type begin,
                                          Index_type len,
                                          Index_type icount) {
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < len) {
    loop_body(ii + icount, ii + begin);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
__global__ void forall_cuda_kernel(LOOP_BODY loop_body,
                                   const Index_type* idx,
                                   Index_type length) {
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
template <typename LOOP_BODY>
__global__ void forall_Icount_cuda_kernel(LOOP_BODY loop_body,
                                          const Index_type* idx,
                                          Index_type length,
                                          Index_type icount) {
  Index_type ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii < length) {
    loop_body(ii + icount, idx[ii]);
  }
}

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over index ranges.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec<BLOCK_SIZE>,
                        Index_type begin,
                        Index_type end,
                        LOOP_BODY loop_body) {
  Index_type len = end - begin;
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, begin, len);
  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range via CUDA kernal launch
 *         without call to cudaDeviceSynchronize() after kernel completes.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec_async<BLOCK_SIZE>,
                        Index_type begin,
                        Index_type end,
                        LOOP_BODY loop_body) {
  Index_type len = end - begin;
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, begin, len);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range with index count,
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec<BLOCK_SIZE>,
                               Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  Index_type len = end - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      begin,
                                                      len,
                                                      icount);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over index range with index count,
 *         via CUDA kernal launch without call to cudaDeviceSynchronize()
 *         after kernel completes.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec_async<BLOCK_SIZE>,
                               Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  Index_type len = end - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      begin,
                                                      len,
                                                      icount);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

#if defined(RAJA_USE_BOXSEGMENT)
//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over Box Segments.
//
////////////////////////////////////////////////////////////////////////
//

template <typename LOOP_BODY>
__global__ void forall_box1d_cuda_kernel(LOOP_BODY loop_body,
                                       Index_type corner,
                                       Index_type xExtent, Index_type xStride)
{
   Index_type xIdx = blockDim.x * blockIdx.x + threadIdx.x ;

   if (xIdx < xExtent) {
      Index_type ii = corner + xIdx*xStride ;
      loop_body(ii);
   }
}

template <typename LOOP_BODY>
__global__ void forall_box1d_Icount_cuda_kernel(LOOP_BODY loop_body,
                                       Index_type corner,
                                       Index_type xExtent, Index_type xStride,
                                       Index_type icount)
{
   Index_type xIdx = blockDim.x * blockIdx.x + threadIdx.x ;

   if (xIdx < xExtent) {
      Index_type i  = icount + xIdx ;
      Index_type ii = corner + xIdx*xStride ;
      loop_body(i, ii);
   }
}

template <typename LOOP_BODY>
__global__ void forall_box2d_cuda_kernel(LOOP_BODY loop_body,
                                       Index_type corner,
                                       Index_type xExtent, Index_type yExtent,
                                       Index_type xStride, Index_type yStride)
{
   Index_type xIdx = blockDim.x * blockIdx.x + threadIdx.x ;
   Index_type yIdx = blockDim.y * blockIdx.y + threadIdx.y ;

   if ((xIdx < xExtent) && (yIdx < yExtent)) {
      Index_type ii = corner + xIdx*xStride + yIdx*yStride ;
      loop_body(ii);
   }
}

template <typename LOOP_BODY>
__global__ void forall_box2d_Icount_cuda_kernel(LOOP_BODY loop_body,
                                       Index_type corner,
                                       Index_type xExtent, Index_type yExtent,
                                       Index_type xStride, Index_type yStride,
                                       Index_type icount)
{
   Index_type xIdx = blockDim.x * blockIdx.x + threadIdx.x ;
   Index_type yIdx = blockDim.y * blockIdx.y + threadIdx.y ;

   if ((xIdx < xExtent) && (yIdx < yExtent)) {
      Index_type i  = icount + yIdx*xExtent + xIdx ;
      Index_type ii = corner + xIdx*xStride + yIdx*yStride ;
      loop_body(i, ii);
   }
}

template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec<BLOCK_SIZE>,
            const BoxSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type  corner = iseg.getCorner();
   Index_type  dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      // dim3 dimBlock(32, 4, 1) ;
      // dim3 dimGrid((xExtent + 31)/32, (yExtent+3)/4, 1) ;
      dim3 dimBlock(BLOCK_SIZE, 1, 1) ;
      dim3 dimGrid((xExtent + BLOCK_SIZE - 1)/BLOCK_SIZE, yExtent, 1) ;

      RAJA_FT_BEGIN ;

      forall_box2d_cuda_kernel<<<dimGrid, dimBlock>>>(loop_body, corner,
                                                      xExtent, yExtent,
                                                      xStride, yStride);
      cudaErrchk(cudaPeekAtLastError());
      cudaErrchk(cudaDeviceSynchronize());

      RAJA_FT_END ;
   }
   else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      size_t gridSize = (xExtent + BLOCK_SIZE - 1) / BLOCK_SIZE;

      RAJA_FT_BEGIN ;

      forall_box1d_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, corner,
                                                         xExtent, xStride) ;
      cudaErrchk(cudaPeekAtLastError());
      cudaErrchk(cudaDeviceSynchronize());

      RAJA_FT_END ;
   }
}

template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE
void forall(cuda_exec_async<BLOCK_SIZE>,
            const BoxSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type  corner = iseg.getCorner();
   Index_type  dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      // dim3 dimBlock(32, 4, 1) ;
      // dim3 dimGrid((xExtent + 31)/32, (yExtent+3)/4, 1) ;
      dim3 dimBlock(BLOCK_SIZE, 1, 1) ;
      dim3 dimGrid((xExtent + BLOCK_SIZE - 1)/BLOCK_SIZE, yExtent, 1) ;

      RAJA_FT_BEGIN ;

      forall_box2d_cuda_kernel<<<dimGrid, dimBlock>>>(loop_body, corner,
                                                      xExtent, yExtent,
                                                      xStride, yStride);
      cudaErrchk(cudaPeekAtLastError());

      RAJA_FT_END ;
   }
   else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      size_t gridSize = (xExtent + BLOCK_SIZE - 1) / BLOCK_SIZE;

      RAJA_FT_BEGIN ;

      forall_box1d_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, corner,
                                                         xExtent, xStride) ;
      cudaErrchk(cudaPeekAtLastError());

      RAJA_FT_END ;
   }
}

template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec<BLOCK_SIZE>,
                   const BoxSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type  corner = iseg.getCorner();
   Index_type  dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      // dim3 dimBlock(32, 4, 1) ;
      // dim3 dimGrid((xExtent + 31)/32, (yExtent+3)/4, 1) ;
      dim3 dimBlock(BLOCK_SIZE, 1, 1) ;
      dim3 dimGrid((xExtent + BLOCK_SIZE - 1)/BLOCK_SIZE, yExtent, 1) ;

      RAJA_FT_BEGIN ;

      forall_box2d_Icount_cuda_kernel<<<dimGrid, dimBlock>>>(loop_body, corner,
                                                             xExtent, yExtent,
                                                             xStride, yStride,
                                                             icount);
      cudaErrchk(cudaPeekAtLastError());
      cudaErrchk(cudaDeviceSynchronize());

      RAJA_FT_END ;
   }
   else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      size_t gridSize = (xExtent + BLOCK_SIZE - 1) / BLOCK_SIZE;

      RAJA_FT_BEGIN ;

      forall_box1d_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                                corner,
                                                                xExtent,
                                                                xStride,
                                                                icount) ;
      cudaErrchk(cudaPeekAtLastError());
      cudaErrchk(cudaDeviceSynchronize());

      RAJA_FT_END ;
   }
}

template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cuda_exec_async<BLOCK_SIZE>,
                   const BoxSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type  corner = iseg.getCorner();
   Index_type  dim    = iseg.getDim();
   Index_type const *extent = iseg.getExtent();
   Index_type const *stride = iseg.getStride();

   if (dim == 2) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;
      Index_type yExtent = extent[1] ; /* slow stride */
      Index_type yStride = stride[1] ;

      // dim3 dimBlock(32, 4, 1) ;
      // dim3 dimGrid((xExtent + 31)/32, (yExtent+3)/4, 1) ;
      dim3 dimBlock(BLOCK_SIZE, 1, 1) ;
      dim3 dimGrid((xExtent + BLOCK_SIZE - 1)/BLOCK_SIZE, yExtent, 1) ;

      RAJA_FT_BEGIN ;

      forall_box2d_Icount_cuda_kernel<<<dimGrid, dimBlock>>>(loop_body, corner,
                                                             xExtent, yExtent,
                                                             xStride, yStride,
                                                             icount);
      cudaErrchk(cudaPeekAtLastError());

      RAJA_FT_END ;
   }
   else if (dim == 1) {
      Index_type xExtent = extent[0] ; /* fast stride */
      Index_type xStride = stride[0] ;

      size_t gridSize = (xExtent + BLOCK_SIZE - 1) / BLOCK_SIZE;

      RAJA_FT_BEGIN ;

      forall_box1d_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                                corner,
                                                                xExtent,
                                                                xStride,
                                                                icount) ;
      cudaErrchk(cudaPeekAtLastError());

      RAJA_FT_END ;
   }
}
#endif  // defined(RAJA_USE_BOXSEGMENT)


//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over range segments.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over range segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec<BLOCK_SIZE>,
                        const RangeSegment& iseg,
                        LOOP_BODY loop_body) {
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();
  Index_type len = end - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, begin, len);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over range segment object via CUDA kernal launch
 *         without call to cudaDeviceSynchronize() after kernel completes.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec_async<BLOCK_SIZE>,
                        const RangeSegment& iseg,
                        LOOP_BODY loop_body) {
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();
  Index_type len = end - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, begin, len);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over range segment object with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec<BLOCK_SIZE>,
                               const RangeSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  Index_type begin = iseg.getBegin();
  Index_type len = iseg.getEnd() - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      begin,
                                                      len,
                                                      icount);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over range segment object with index count
 *         via CUDA kernal launch without call to cudaDeviceSynchronize()
 *         after kernel completes.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec_async<BLOCK_SIZE>,
                               const RangeSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  Index_type begin = iseg.getBegin();
  Index_type len = iseg.getEnd() - begin;

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      begin,
                                                      len,
                                                      icount);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

//
////////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for indirection array via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec<BLOCK_SIZE>,
                        const Index_type* idx,
                        Index_type len,
                        LOOP_BODY loop_body) {
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, idx, len);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for indirection array via CUDA kernal launch
 *         without call to cudaDeviceSynchronize() after kernel completes.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec_async<BLOCK_SIZE>,
                        const Index_type* idx,
                        Index_type len,
                        LOOP_BODY loop_body) {
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, idx, len);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over indirection array with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec<BLOCK_SIZE>,
                               const Index_type* idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      idx,
                                                      len,
                                                      icount);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over indirection array with index count
 *         via CUDA kernal launch without call to cudaDeviceSynchronize()
 *         after kernel completes.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec_async<BLOCK_SIZE>,
                               const Index_type* idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      idx,
                                                      len,
                                                      icount);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

//
////////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segments.
//
////////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for list segment object via CUDA kernal launch.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec<BLOCK_SIZE>,
                        const ListSegment& iseg,
                        LOOP_BODY loop_body) {
  const Index_type* idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, idx, len);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution for list segment object via CUDA kernal launch
 *         without call to cudaDeviceSynchronize() after kernel completes.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall(cuda_exec_async<BLOCK_SIZE>,
                        const ListSegment& iseg,
                        LOOP_BODY loop_body) {
  const Index_type* idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body, idx, len);
  cudaErrchk(cudaPeekAtLastError());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over list segment object with index count
 *         via CUDA kernal launch.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec<BLOCK_SIZE>,
                               const ListSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  const Index_type* idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      idx,
                                                      len,
                                                      icount);

  cudaErrchk(cudaPeekAtLastError());
  cudaErrchk(cudaDeviceSynchronize());

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  Forall execution over list segment object with index count
 *         via CUDA kernal launch without call to cudaDeviceSynchronize()
 *         after kernel completes.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(cuda_exec_async<BLOCK_SIZE>,
                               const ListSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body) {
  const Index_type* idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  size_t gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RAJA_FT_BEGIN;

  forall_Icount_cuda_kernel<<<gridSize, BLOCK_SIZE>>>(loop_body,
                                                      idx,
                                                      len,
                                                      icount);
  cudaErrchk(cudaPeekAtLastError());

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
                        LOOP_BODY loop_body) {
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
    LOOP_BODY loop_body) {
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
