/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
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

#ifndef RAJA_policy_openmp_nested_HPP
#define RAJA_policy_openmp_nested_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <cassert>
#include <climits>

#include "RAJA/RAJA.hpp"
#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"


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

    using data_t = camp::decay<Data>;

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);

#pragma omp parallel
    {
      data_t private_data = data;

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
        }
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

    using data_t = camp::decay<Data>;

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);
    auto l2 = segment_length<Arg2>(data);

#pragma omp parallel
    {
      data_t private_data = data;

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for collapse(3)
#else
#pragma omp for
#endif
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          for (auto i2 = (decltype(l2))0; i2 < l2; ++i2) {
            private_data.template assign_offset<Arg0>(i0);
            private_data.template assign_offset<Arg1>(i1);
            private_data.template assign_offset<Arg2>(i2);
            execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
          }
        }
      }
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
