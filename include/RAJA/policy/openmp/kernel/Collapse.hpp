/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *
 ******************************************************************************
 */

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

#ifndef RAJA_policy_openmp_nested_HPP
#define RAJA_policy_openmp_nested_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <cassert>
#include <climits>

#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"

#if !defined(RAJA_COMPILER_MSVC)
#define RAJA_COLLAPSE(X) collapse(X)
#else
#define RAJA_COLLAPSE(X)
#endif

namespace RAJA
{

struct omp_parallel_collapse_exec
    : make_policy_pattern_t<RAJA::Policy::openmp,
                            RAJA::Pattern::forall,
                            RAJA::policy::omp::For> {
};

namespace internal
{

/////////
// Collapsing two loops
/////////

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    // NOTE: these are here to avoid a use-after-scope detected by address
    // sanitizer, probably a false positive, but the result should be
    // essentially identical
    auto i0 = l0;
    auto i1 = l1;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp parallel for private(i0, i1) firstprivate(privatizer) \
    RAJA_COLLAPSE(2)
    for (i0 = 0; i0 < l0; ++i0) {
      for (i1 = 0; i1 < l1; ++i1) {
        auto& private_data = privatizer.get_priv();
        private_data.template assign_offset<Arg0>(i0);
        private_data.template assign_offset<Arg1>(i1);
        execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
      }
    }
  }
};


template <camp::idx_t Arg0,
          camp::idx_t Arg1,
          camp::idx_t Arg2,
          typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    const auto l2 = segment_length<Arg2>(data);
    auto i0 = l0;
    auto i1 = l1;
    auto i2 = l2;

    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(data);
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer) \
    RAJA_COLLAPSE(3)
    for (i0 = 0; i0 < l0; ++i0) {
      for (i1 = 0; i1 < l1; ++i1) {
        for (i2 = 0; i2 < l2; ++i2) {
          auto& private_data = privatizer.get_priv();
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          private_data.template assign_offset<Arg2>(i2);
          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
        }
      }
    }
  }
};





}  // namespace internal
}  // namespace RAJA

#undef RAJA_COLLAPSE

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
