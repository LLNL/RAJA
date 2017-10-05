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

#ifndef RAJA_forall_openmp_HPP
#define RAJA_forall_openmp_HPP

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

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/index/Graph.hpp"
#include "RAJA/index/GraphStorage.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include <iostream>
#include <thread>

#include <omp.h>

namespace RAJA
{

namespace impl
{
///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func, typename InnerPolicy>
RAJA_INLINE void forall(const omp_parallel_exec<InnerPolicy>&,
                        Iterable&& iter,
                        Func&& loop_body)
{
#pragma omp parallel
  {
    typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename IndexType,
          typename Func,
          typename InnerPolicy>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const omp_parallel_exec<InnerPolicy>&,
              Iterable&& iter,
              IndexType icount,
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
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const omp_for_nowait_exec&,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for nowait
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(static_cast<IndexType>(i + icount), begin[i]);
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
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const omp_for_exec&,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(static_cast<IndexType>(i + icount), begin[i]);
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
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Iterable,
          typename IndexType,
          typename Func,
          size_t ChunkSize>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const omp_for_static<ChunkSize>&,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp for schedule(static, ChunkSize)
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(static_cast<IndexType>(i + icount), begin[i]);
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


template <typename Iterable, typename Func>
RAJA_INLINE void forall(const omp_for_dependence_graph&,
                        Iterable&& iter,
                        Func&& loop_body)
{

  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  GraphStorageRange storage(iter);

#pragma omp parallel for schedule(static, 1)
  for (decltype(distance) i = 0; i < distance; ++i) {
    storage.wait(begin[i]);  //task->wait()
    loop_body(begin[i]);
    storage.completed(begin[i]);
  }//end iterate over segments of index set

  //storage.reset(); //task->reset() for all tasks
}


template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(const omp_for_dependence_graph&,
              Iterable&& iter,
              IndexType icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  GraphStorageRange storage(iter);

#pragma omp parallel for schedule(static, 1)
  for (decltype(distance) i = 0; i < distance; ++i) {
    storage.wait(begin[i]);  //task->wait()
    loop_body(static_cast<IndexType>(i + icount), begin[i]);
    storage.completed(begin[i]);
  }//end iterate over segments of index set

  //storage.reset(); //task->reset() for all tasks
}


}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard

#include "RAJA/policy/openmp/target_forall.hpp"
