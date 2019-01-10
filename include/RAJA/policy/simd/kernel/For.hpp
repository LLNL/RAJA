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


#ifndef RAJA_policy_simd_kernel_For_HPP
#define RAJA_policy_simd_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{

namespace internal
{


/*!
 *
 * Helper structs to detect lambdas
 *
 */
template <class T>
struct TypeIsLambda {
  static const bool value = false;
};

template <camp::idx_t BodyIdx>
struct TypeIsLambda<RAJA::statement::Lambda<BodyIdx>> {
  static const bool value = true;
};

/*!
 *
 *  Helper structs to invoke a chain of lambdas
 *
 */
template <camp::idx_t LoopIdx, class... States>
struct Invoke_all_Lambda {

  template <camp::idx_t... OffsetIdx,
            camp::idx_t... ParamIdx,
            typename Data,
            typename Offs,
            typename Params>
  static RAJA_INLINE void lambda_special(camp::idx_seq<OffsetIdx...> const &,
                                         camp::idx_seq<ParamIdx...> const &,
                                         Data &,
                                         Offs const &,
                                         Params const &)
  {
  }
};

template <camp::idx_t LoopIdx>
struct Invoke_all_Lambda<LoopIdx> {

  static const bool value = true;

  template <camp::idx_t... OffsetIdx,
            camp::idx_t... ParamIdx,
            typename Data,
            typename Offs,
            typename Params>
  static RAJA_INLINE void lambda_special(camp::idx_seq<OffsetIdx...> const &,
                                         camp::idx_seq<ParamIdx...> const &,
                                         Data &,
                                         Offs const &,
                                         Params const &)
  {
  }
};

template <camp::idx_t LoopIdx, class State, class... States>
struct Invoke_all_Lambda<LoopIdx, State, States...>
    : Invoke_all_Lambda<LoopIdx, States...> {

  // Lambda check
  static const bool value = TypeIsLambda<camp::decay<State>>::value;
  static_assert(value, "Lambdas are only supported post RAJA::simd_exec");

  // Invoke the chain of lambdas
  template <camp::idx_t... OffsetIdx,
            camp::idx_t... ParamIdx,
            typename Data,
            typename Offs,
            typename Params>
  static RAJA_INLINE void lambda_special(camp::idx_seq<OffsetIdx...> const &,
                                         camp::idx_seq<ParamIdx...> const &,
                                         Data &data,
                                         Offs const &offset_tuple,
                                         Params const &params)
  {
    camp::get<LoopIdx>(
        data.bodies)((camp::get<OffsetIdx>(data.segment_tuple)
                          .begin()[camp::get<OffsetIdx>(offset_tuple)])...,
                     camp::get<ParamIdx>(data.param_tuple)...);

    Invoke_all_Lambda<LoopIdx + 1, States...>::lambda_special(
        camp::idx_seq_from_t<decltype(offset_tuple)>{},
        camp::idx_seq_from_t<decltype(params)>{},
        data,
        offset_tuple,
        params);
  }
};


/*!
 * RAJA::kernel forall_impl executor specialization for statement::For.
 * Assumptions: RAJA::simd_exec is the inner most policy,
 * only one lambda is used, no reductions are done within the lambda.
 * Assigns the loop index to offset ArgumentId
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct StatementExecutor<
    statement::For<ArgumentId, RAJA::simd_exec, EnclosedStmts...>> {

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
      auto offsets = data.offset_tuple;
      auto params = data.param_tuple;
      get<ArgumentId>(offsets) = i;

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


#endif /* RAJA_pattern_nested_HPP */
