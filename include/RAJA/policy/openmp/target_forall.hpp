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

#ifndef RAJA_target_forall_openmp_HXX
#define RAJA_target_forall_openmp_HXX

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include <omp.h>

namespace RAJA
{

namespace policy
{

namespace omp
{

///
/// OpenMP target parallel for policy implementation
///

template <size_t Teams, typename Iterable, typename Func>
// RAJA_INLINE void forall(const omp_target_parallel_for_exec<Teams>&,
RAJA_INLINE void forall_impl(const omp_target_parallel_for_exec<Teams>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma omp target teams distribute parallel for num_teams(Teams) \
    schedule(static, 1) map(to : body)
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
#pragma omp target teams distribute parallel for schedule(static, \
                                                          1) map(to : body)
  for (Index_type i = 0; i < distance; ++i) {
    Body ib = body;
    ib(begin[i]);
  }
}


}  // closing brace for omp namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_TARGET_RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
