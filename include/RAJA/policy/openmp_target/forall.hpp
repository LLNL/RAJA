//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_target_forall_openmp_HPP
#define RAJA_target_forall_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{

namespace policy
{

namespace omp
{

///
/// OpenMP target parallel for policy implementation
///

template <size_t ThreadsPerTeam, typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Omp>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate<RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
  >
forall_impl(resources::Omp omp_res,
            const omp_target_parallel_for_exec<ThreadsPerTeam>& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  using EXEC_POL = typename std::decay<decltype(p)>::type;
  RAJA::expt::ParamMultiplexer::init<EXEC_POL>(f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;

  RAJA_EXTRACT_BED_IT(iter);

  // Reset if exceed CUDA threads per block limit.
  int tperteam = ThreadsPerTeam;
  if ( tperteam > omp::MAXNUMTHREADS )
  {
    tperteam = omp::MAXNUMTHREADS;
  }

  // calculate number of teams based on user defined threads per team
  // datasize is distance between begin() and end() of iterable
  auto numteams = RAJA_DIVIDE_CEILING_INT( distance_it, tperteam );
  if ( numteams > tperteam )
  {
    // Omp target reducers will write team # results, into Threads-sized array.
    // Need to insure NumTeams <= Threads to prevent array out of bounds access.
    numteams = tperteam;
  }

// thread_limit(tperteam) unused due to XL seg fault (when tperteam != distance)
  auto i = distance_it;

#pragma omp target teams distribute parallel for num_teams(numteams) \
    schedule(static, 1) map(to : body,begin_it) reduction(combine: f_params)
  for (i = 0; i < distance_it; ++i) {
    Body ib = body;
    RAJA::expt::invoke_body(f_params, ib, begin_it[i]);
  }

  RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params);
  return resources::EventProxy<resources::Omp>(omp_res);
}

template <size_t ThreadsPerTeam, typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Omp>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>
  >
forall_impl(resources::Omp omp_res,
            const omp_target_parallel_for_exec<ThreadsPerTeam>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;

  RAJA_EXTRACT_BED_IT(iter);

  // Reset if exceed CUDA threads per block limit.
  int tperteam = ThreadsPerTeam;
  if ( tperteam > omp::MAXNUMTHREADS )
  {
    tperteam = omp::MAXNUMTHREADS;
  }

  // calculate number of teams based on user defined threads per team
  // datasize is distance between begin() and end() of iterable
  auto numteams = RAJA_DIVIDE_CEILING_INT( distance_it, tperteam );
  if ( numteams > tperteam )
  {
    // Omp target reducers will write team # results, into Threads-sized array.
    // Need to insure NumTeams <= Threads to prevent array out of bounds access.
    numteams = tperteam;
  }

// thread_limit(tperteam) unused due to XL seg fault (when tperteam != distance)
  auto i = distance_it;

#pragma omp target teams distribute parallel for num_teams(numteams) \
    schedule(static, 1) map(to : body,begin_it)
  for (i = 0; i < distance_it; ++i) {
    Body ib = body;
    ib(begin_it[i]);
  }

  return resources::EventProxy<resources::Omp>(omp_res);
}





template <typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE
concepts::enable_if_t<
  resources::EventProxy<resources::Omp>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate<RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
  >
forall_impl(resources::Omp omp_res,
            const omp_target_parallel_for_exec_nt& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  using EXEC_POL = typename std::decay<decltype(p)>::type;
  RAJA::expt::ParamMultiplexer::init<EXEC_POL>(f_params);
  RAJA_OMP_DECLARE_REDUCTION_COMBINE;

  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;

  RAJA_EXTRACT_BED_IT(iter);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    firstprivate(body,begin_it) reduction(combine: f_params)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    Body ib = body;
    RAJA::expt::invoke_body(f_params, ib, begin_it[i]);
  }

  RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params);
  return resources::EventProxy<resources::Omp>(omp_res);
}

template <typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE
concepts::enable_if_t<
  resources::EventProxy<resources::Omp>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>
  >
forall_impl(resources::Omp omp_res,
            const omp_target_parallel_for_exec_nt&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;

  RAJA_EXTRACT_BED_IT(iter);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    firstprivate(body,begin_it)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    Body ib = body;
    ib(begin_it[i]);
  }

  return resources::EventProxy<resources::Omp>(omp_res);
}

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_TARGET_RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
