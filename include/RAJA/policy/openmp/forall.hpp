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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_openmp_HPP
#define RAJA_forall_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <iostream>
#include <type_traits>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/index/Graph.hpp"
#include "RAJA/index/GraphStorage.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"


namespace RAJA
{

namespace policy
{
namespace omp
{
///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func, typename InnerPolicy>
RAJA_INLINE void forall_impl(const omp_parallel_exec<InnerPolicy>&,
                             Iterable&& iter,
                             Func&& loop_body)
{

  RAJA::region<RAJA::omp_parallel_region>([&]() {
    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    forall_impl(InnerPolicy{}, iter, body.get_priv());
  });
}

///
/// OpenMP for nowait policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_for_nowait_exec&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for nowait
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }
}

///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_for_exec&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }
}

///
/// OpenMP parallel for static policy implementation
///

template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall_impl(const omp_for_static<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(static, ChunkSize)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
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
RAJA_INLINE void forall_impl(const omp_for_dependence_graph&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);

  //auto begin = std::begin(iter);
  //auto end = std::end(iter);
  //auto distance = std::distance(begin, end);

  GraphStorageRange storage(iter);

#pragma omp parallel for schedule(static, 1)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    storage.wait(begin_it[i]);  //task->wait()
    loop_body(begin_it[i]);
    storage.completed(begin_it[i]);
  }//end iterate over segments of index set

  //storage.reset(); //task->reset() for all tasks
}


template <typename Iterable, typename IndexType, typename Func>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount_impl(const omp_for_dependence_graph&,
                   Iterable&& iter,
                   IndexType icount,
                   Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);

  //auto begin = std::begin(iter);
  //auto end = std::end(iter);
  //auto distance = std::distance(begin, end);

  GraphStorageRange storage(iter);

#pragma omp parallel for schedule(static, 1)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    storage.wait(begin_it[i]);  //task->wait()
    loop_body(static_cast<IndexType>(i + icount), begin_it[i]);
    storage.completed(begin_it[i]);
  }//end iterate over segments of index set

  //storage.reset(); //task->reset() for all tasks
}


}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
