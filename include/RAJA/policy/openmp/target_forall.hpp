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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_target_forall_openmp_HPP
#define RAJA_target_forall_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace policy
{

namespace omp
{

///
/// OpenMP target parallel for policy implementation
///

template <size_t Threads, typename Iterable, typename Func>
// RAJA_INLINE void forall(const omp_target_parallel_for_exec<Teams>&,
RAJA_INLINE void forall_impl(const omp_target_parallel_for_exec<Threads>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  auto teamnum = RAJA_DIVIDE_CEILING_INT( (int)distance, (int)Threads );
  if ( teamnum > Threads )
  {
    // Omp target reducers will write team # results, into Threads-sized array.
    // Need to insure teams <= Threads to prevent array out of bounds access.
    teamnum = Threads;
  }
#pragma omp target teams distribute parallel for num_teams(teamnum) \
    thread_limit(Threads) schedule(static, 1) map(to              \
                                                    : body)
  for (Index_type i = 0; i < distance; ++i) {
    Body ib = body;
    ib(begin[i]);
  }
}

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_target_parallel_for_exec_nt&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp target teams distribute parallel for schedule(static, 1) \
    map(to                                                           \
        : body)
  for (Index_type i = 0; i < distance; ++i) {
    Body ib = body;
    ib(begin[i]);
  }
}


}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_TARGET_RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
