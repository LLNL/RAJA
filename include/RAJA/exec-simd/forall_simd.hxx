/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_simd_HXX
#define RAJA_forall_simd_HXX

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

#include "RAJA/config.hxx"

#include "RAJA/int_datatypes.hxx"

#include "RAJA/fault_tolerance.hxx"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec,
                        Index_type begin,
                        Index_type end,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = end - begin;

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, ii + begin);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range segments.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec,
                        const RangeSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range set object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               const RangeSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type loop_end = iseg.getEnd() - iseg.getBegin();

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, ii + begin);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec,
                        Index_type begin,
                        Index_type end,
                        Index_type stride,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with stride with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type stride,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = (end - begin) / stride;
  if ((end - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, begin + ii * stride);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range-stride segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range segment object with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec,
                        const RangeStrideSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();
  Index_type stride = iseg.getStride();

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range index set with stride object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               const RangeStrideSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type stride = iseg.getStride();
  Index_type loop_end = (iseg.getEnd() - begin) / stride;
  if ((iseg.getEnd() - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

  RAJA_SIMD
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, begin + ii * stride);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
// NOTE: These operations will not vectorize. We include them here and
//       force sequential execution for convenience.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec,
                        const Index_type* RAJA_RESTRICT idx,
                        Index_type len,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over indices in indirection array
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               const Index_type* RAJA_RESTRICT idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segment objects.
//
// NOTE: These operations will not vectorize. We include them here and
//       force sequential execution for convenience.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(simd_exec, const ListSegment& iseg, LOOP_BODY loop_body)
{
  const Index_type* RAJA_RESTRICT idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(simd_exec,
                               const ListSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  const Index_type* RAJA_RESTRICT idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// SIMD execution policy does not apply to iteration over index
// set segments, only to execution of individual segments. So there
// are no index set traversal methods in this file.
//
//////////////////////////////////////////////////////////////////////
//

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
