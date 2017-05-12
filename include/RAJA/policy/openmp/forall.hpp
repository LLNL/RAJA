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

#include "RAJA/config.hpp"

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

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/IndexSet.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include <iostream>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif


namespace RAJA
{

namespace impl
{
///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename InnerPolicy, typename Func>
RAJA_INLINE void forall(const omp_parallel_exec<InnerPolicy>&,
                        Iterable&& iter,
                        Func&& loop_body)
{
#pragma omp parallel
  {
    typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
    forall<InnerPolicy>(std::forward<Iterable>(iter),
                        std::forward<Func>(body));
  }
}

template <typename Iterable, typename InnerPolicy, typename Func>
RAJA_INLINE void forall_Icount(const omp_parallel_exec<InnerPolicy>&,
                               Iterable&& iter,
                               Index_type icount,
                               Func&& loop_body)
{
#pragma omp parallel
  {
    typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

///
/// OpenMP for nowait policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const omp_for_nowait_exec&,
                        Iterable&& iter,
                        Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for nowait
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_Icount(const omp_for_nowait_exec&,
                               Iterable&& iter,
                               Index_type icount,
                               Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for nowait
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall(const omp_for_exec&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_Icount(const omp_for_exec&,
                               Iterable&& iter,
                               Index_type icount,
                               Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

///
/// OpenMP parallel for static policy implementation
///

template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall(const omp_for_static<ChunkSize>&,
                        Iterable&& iter,
                        Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for schedule(static, ChunkSize)
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall_Icount(const omp_for_static<ChunkSize>&,
                               Iterable&& iter,
                               Index_type icount,
                               Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for schedule(static, ChunkSize)
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
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
 * \brief  Iterate over index set segments using an omp parallel loop and
 *         segment dependency graph. Individual segment execution will use
 *         execution policy template parameter.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */

    /*
    * TODO: Fix this!!!
     */

/*
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY, typename ... SEG_TYPES>
RAJA_INLINE void forall(
    ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
    const IndexSet<SEG_TYPES ...>& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    RAJA_ABORT_OR_THROW("IndexSet dependency graph");
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));

  int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
    DepGraphNode* task = seg_info->getDepGraphNode();

    task->wait();

    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

    task->reset();

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
        dep->satisfyOne();
      }
    }

  }  // iterate over segments of index set
}
*/

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
