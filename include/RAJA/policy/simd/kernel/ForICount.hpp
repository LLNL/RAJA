/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for statement wrappers and executors.
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


#ifndef RAJA_policy_simd_kernel_ForICount_HPP
#define RAJA_policy_simd_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/simd/policy.hpp"
#include "RAJA/policy/simd/kernel/For.hpp"

namespace RAJA
{

namespace internal
{


/*!
 * RAJA::kernel forall_impl executor specialization for statement::ForICount.
 * Assumptions: RAJA::simd_exec is the inner most policy,
 * only one lambda is used, no reductions are done within the lambda.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <camp::idx_t ArgumentId, typename ParamId,
          typename... EnclosedStmts>
struct StatementExecutor<
    statement::ForICount<ArgumentId, ParamId, RAJA::simd_exec,
                         EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    auto iter = get<ArgumentId>(data.segment_tuple);
    auto begin = std::begin(iter);
    auto end = std::end(iter);
    auto distance = std::distance(begin, end);

    RAJA_SIMD
    for (decltype(distance) i = 0; i < distance; ++i) {

      // Offsets and parameters need to be privatized
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      auto offsets = data.offset_tuple;
      auto params = data.param_tuple;
      Invoke_all_Lambda<0, EnclosedStmts...>::lambda_special(
          camp::idx_seq_from_t<decltype(offsets)>{},
          camp::idx_seq_from_t<decltype(params)>{},
          data,
          offsets,
          params);
    }
  }
};

}  // namespace internal
}  // end namespace RAJA


#endif 
