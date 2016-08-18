/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for OpenMP.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_openmp_HXX
#define RAJA_forall_openmp_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_OPENMP)

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

#include "RAJA/segment_exec.hxx"

#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
 * \brief  omp parallel for iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        Index_type begin,
                        Index_type end,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static) firstprivate(loop_body)
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        Index_type begin,
                        Index_type end,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = end - begin;

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static)
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, ii + begin);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = end - begin;

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
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
 * \brief  omp parallel for iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        const RangeSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static) firstprivate(loop_body)
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        const RangeSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
  for (Index_type ii = begin; ii < end; ++ii) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over range segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               const RangeSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type loop_end = iseg.getEnd() - begin;

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static) firstprivate(loop_body)
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, ii + begin);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over range segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               const RangeSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type loop_end = iseg.getEnd() - begin;

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
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
 * \brief  omp parallel for iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        Index_type begin,
                        Index_type end,
                        Index_type stride,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static)
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        Index_type begin,
                        Index_type end,
                        Index_type stride,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range with stride
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type stride,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = (end - begin) / stride;
  if ((end - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static)
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, begin + ii * stride);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over index range with stride
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               Index_type begin,
                               Index_type end,
                               Index_type stride,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type loop_end = (end - begin) / stride;
  if ((end - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
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
 * \brief  omp parallel for iteration over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        const RangeStrideSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();
  Index_type stride = iseg.getStride();

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static)
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        const RangeStrideSegment& iseg,
                        LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type end = iseg.getEnd();
  Index_type stride = iseg.getStride();

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
  for (Index_type ii = begin; ii < end; ii += stride) {
    loop_body(ii);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over range-stride segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               const RangeStrideSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type stride = iseg.getStride();
  Index_type loop_end = (iseg.getEnd() - begin) / stride;
  if ((iseg.getEnd() - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

#pragma omp parallel for schedule(static)
  for (Index_type ii = 0; ii < loop_end; ++ii) {
    loop_body(ii + icount, begin + ii * stride);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over range-stride segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               const RangeStrideSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  Index_type begin = iseg.getBegin();
  Index_type stride = iseg.getStride();
  Index_type loop_end = (iseg.getEnd() - begin) / stride;
  if ((iseg.getEnd() - begin) % stride != 0) loop_end++;

  RAJA_FT_BEGIN;

#pragma omp for schedule(static) nowait
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
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        const Index_type* __restrict__ idx,
                        Index_type len,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
#pragma omp parallel for schedule(static)
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        const Index_type* __restrict__ idx,
                        Index_type len,
                        LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
#pragma omp for schedule(static) nowait
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over indirection array
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               const Index_type* __restrict__ idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
#pragma omp parallel for schedule(static)
  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over indirection array
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               const Index_type* __restrict__ idx,
                               Index_type len,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  RAJA_FT_BEGIN;

#pragma novector
#pragma omp for schedule(static) nowait
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
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_parallel_for_exec,
                        const ListSegment& iseg,
                        LOOP_BODY loop_body)
{
  const Index_type* __restrict__ idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
#pragma omp parallel for schedule(static)
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall(omp_for_nowait_exec,
                        const ListSegment& iseg,
                        LOOP_BODY loop_body)
{
  const Index_type* __restrict__ idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
#pragma omp for schedule(static) nowait
  for (Index_type k = 0; k < len; ++k) {
    loop_body(idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over list segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_parallel_for_exec,
                               const ListSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  const Index_type* __restrict__ idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
#pragma omp parallel for schedule(static)
  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

/*!
 ******************************************************************************
 *
 * \brief  omp for "nowait" iteration over list segment object
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_Icount(omp_for_nowait_exec,
                               const ListSegment& iseg,
                               Index_type icount,
                               LOOP_BODY loop_body)
{
  const Index_type* __restrict__ idx = iseg.getIndex();
  Index_type len = iseg.getLength();

  RAJA_FT_BEGIN;

#pragma novector
#pragma omp for schedule(static) nowait
  for (Index_type k = 0; k < len; ++k) {
    loop_body(k + icount, idx[k]);
  }

  RAJA_FT_END;
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using omp execution. Segment execution is defined by
// segment execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using omp parallel for
 *         and use execution policy template parameter for segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  int num_seg = iset.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

  }  // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using an omp parallel loop and
 *         segment dependency graph. Individual segment execution will use
 *         execution policy template parameter.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall(
    IndexSet::ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));

  int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
    DepGraphNode* task = seg_info->getDepGraphNode();

    //
    // This is declared volatile to prevent compiler from
    // optimizing the while loop (into an if-statement, for example).
    // It may not be able to see that the value accessed through
    // the method call will be changed at the end of the for-loop
    // from another executing thread.
    //
    volatile int* __restrict__ semVal = &(task->semaphoreValue());

    while (*semVal != 0) {
      /* spin or (better) sleep here */;
      // printf("%d ", *semVal) ;
      // sleep(1) ;
      // for (volatile int spin = 0; spin<1000; ++spin) {
      //    spin = spin ;
      // }
      sched_yield();
    }

    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

    if (task->semaphoreReloadValue() != 0) {
      task->semaphoreValue() = task->semaphoreReloadValue();
    }

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
        __sync_fetch_and_sub(&(dep->semaphoreValue()), 1);
      }
    }

  }  // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using omp parallel for
 *         execution and use execution policy template parameter for segments.
 *
 *         This method passes index count to segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY>
RAJA_INLINE void forall_Icount(
    IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
    const IndexSet& iset,
    LOOP_BODY loop_body)
{
  int num_seg = iset.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
    executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(seg_info, loop_body);

  }  // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Special segment iteration using OpenMP parallel region around
 *         segment iteration loop. Individual segment execution is defined
 *         in loop body.
 *
 *         This method does not use a task dependency graph for
 *         the index set segments.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_segments(omp_parallel_segit,
                                 const IndexSet& iset,
                                 LOOP_BODY loop_body)
{
  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));
  int num_seg = ncis.getNumSegments();

#pragma omp parallel
  {
    int numThreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    /* Create a temporary IndexSet with one Segment */
    IndexSet is_tmp;
    is_tmp.push_back(RangeSegment(0, 0));  // create a dummy range segment

    RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

    for (int isi = tid; isi < num_seg; isi += numThreads) {
      RangeSegment* isetSeg = static_cast<RangeSegment*>(ncis.getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin());
      segTmp->setEnd(isetSeg->getEnd());
      segTmp->setPrivate(isetSeg->getPrivate());

      loop_body(&is_tmp);

    }  // loop over index set segments

  }  // end omp parallel region
}

/*!
 ******************************************************************************
 *
 * \brief  Special task-graph segment iteration using OpenMP parallel region
 *         around segment iteration loop and explicit task dependency graph.
 *         Individual segment execution is defined in loop body.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_segments(omp_taskgraph_segit,
                                 const IndexSet& iset,
                                 LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));
  int num_seg = ncis.getNumSegments();

#pragma omp parallel
  {
    int numThreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    /* Create a temporary IndexSet with one Segment */
    IndexSet is_tmp;
    is_tmp.push_back(RangeSegment(0, 0));  // create a dummy range segment

    RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

    for (int isi = tid; isi < num_seg; isi += numThreads) {
      IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
      DepGraphNode* task = seg_info->getDepGraphNode();

      //
      // This is declared volatile to prevent compiler from
      // optimizing the while loop (into an if-statement, for example).
      // It may not be able to see that the value accessed through
      // the method call will be changed at the end of the for-loop
      // from another executing thread.
      //
      volatile int* __restrict__ semVal = &(task->semaphoreValue());

      while (*semVal != 0) {
        /* spin or (better) sleep here */;
        // printf("%d ", *semVal) ;
        // sleep(1) ;
        // volatile int spin ;
        // for (spin = 0; spin<1000; ++spin) {
        //    spin = spin ;
        // }
        sched_yield();
      }

      RangeSegment* isetSeg = static_cast<RangeSegment*>(ncis.getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin());
      segTmp->setEnd(isetSeg->getEnd());
      segTmp->setPrivate(isetSeg->getPrivate());

      loop_body(&is_tmp);

      if (task->semaphoreReloadValue() != 0) {
        task->semaphoreValue() = task->semaphoreReloadValue();
      }

      if (task->numDepTasks() != 0) {
        for (int ii = 0; ii < task->numDepTasks(); ++ii) {
          // Alternateively, we could get the return value of this call
          // and actively launch the task if we are the last depedent
          // task. In that case, we would not need the semaphore spin
          // loop above.
          int seg = task->depTaskNum(ii);
          DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
          __sync_fetch_and_sub(&(dep->semaphoreValue()), 1);
        }
      }

    }  // loop over index set segments

  }  // end omp parallel region
}

/*!
 ******************************************************************************
 *
 * \brief  Special task-graph segment iteration using OpenMP parallel region
 *         around segment iteration loop and explicit task dependency graph.
 *         Individual segment execution is defined in loop body.
 *
 *         This method differs from the preceding one in that this one
 *         has each OpenMP thread working on a set of segments defined by a
 *         contiguous interval of segment ids in the index set.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set. It also
 *         assumes that the segment interval for each thread has been defined.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE void forall_segments(omp_taskgraph_interval_segit,
                                 const IndexSet& iset,
                                 LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    exit(1);
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));
  int num_seg = ncis.getNumSegments();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    /* Create a temporary IndexSet with one Segment */
    IndexSet is_tmp;
    is_tmp.push_back(RangeSegment(0, 0));  // create a dummy range segment

    RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

    const int tbegin = ncis.getSegmentIntervalBegin(tid);
    const int tend = ncis.getSegmentIntervalEnd(tid);

    for (int isi = tbegin; isi < tend; ++isi) {
      IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
      DepGraphNode* task = seg_info->getDepGraphNode();

      //
      // This is declared volatile to prevent compiler from
      // optimizing the while loop (into an if-statement, for example).
      // It may not be able to see that the value accessed through
      // the method call will be changed at the end of the for-loop
      // from another executing thread.
      //
      volatile int* __restrict__ semVal = &(task->semaphoreValue());

      while (*semVal != 0) {
        /* spin or (better) sleep here */;
        // printf("%d ", *semVal) ;
        // sleep(1) ;
        // volatile int spin ;
        // for (spin = 0; spin<1000; ++spin) {
        //    spin = spin ;
        // }
        sched_yield();
      }

      RangeSegment* isetSeg = static_cast<RangeSegment*>(ncis.getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin());
      segTmp->setEnd(isetSeg->getEnd());
      segTmp->setPrivate(isetSeg->getPrivate());

      loop_body(&is_tmp);

      if (task->semaphoreReloadValue() != 0) {
        task->semaphoreValue() = task->semaphoreReloadValue();
      }

      if (task->numDepTasks() != 0) {
        for (int ii = 0; ii < task->numDepTasks(); ++ii) {
          // Alternateively, we could get the return value of this call
          // and actively launch the task if we are the last depedent
          // task. In that case, we would not need the semaphore spin
          // loop above.
          int seg = task->depTaskNum(ii);
          DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
          __sync_fetch_and_sub(&(dep->semaphoreValue()), 1);
        }
      }

    }  // loop over interval segments

  }  // end omp parallel region
}

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
