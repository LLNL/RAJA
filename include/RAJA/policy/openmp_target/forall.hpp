//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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

template <size_t ThreadsPerTeam, typename Iterable, typename Func>
// RAJA_INLINE void forall(const omp_target_parallel_for_exec<Teams>&,
RAJA_INLINE void forall_impl(const omp_target_parallel_for_exec<ThreadsPerTeam>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using Body = typename std::remove_reference<decltype(loop_body)>::type;
  Body body = loop_body;
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  // Reset if exceed CUDA threads per block limit.
  int tperteam = ThreadsPerTeam;
  if ( tperteam > omp::MAXNUMTHREADS )
  {
    tperteam = omp::MAXNUMTHREADS;
  }

  // calculate number of teams based on user defined threads per team
  // datasize is distance between begin() and end() of iterable
  auto numteams = RAJA_DIVIDE_CEILING_INT( distance, tperteam );
  if ( numteams > tperteam )
  {
    // Omp target reducers will write team # results, into Threads-sized array.
    // Need to insure NumTeams <= Threads to prevent array out of bounds access.
    numteams = tperteam;
  }

#pragma omp target teams distribute parallel for num_teams(numteams) \
    thread_limit(tperteam) schedule(static, 1) map(to              \
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
