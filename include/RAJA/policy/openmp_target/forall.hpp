//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

//std::cout << "Mapping body to device..." << std::endl;
//std::cout << typeid(body).name() << std::endl;

//printf("%p\n", &body);

#pragma omp target teams distribute parallel for num_teams(Teams) \
    schedule(static, 1) map(to                                    \
                            : body)
  for (Index_type i = 0; i < distance; ++i) {
//  printf("Running index %d\n", i);
//  printf("%p\n", &body);
    Body ib = body;
//  printf("%p\n", &ib);
//  printf("%d\n", begin[i]);
    ib(begin[i]);
//  printf("Ran index %d\n", i);
  }

//std::cout << "Done on device" << std::endl;
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
