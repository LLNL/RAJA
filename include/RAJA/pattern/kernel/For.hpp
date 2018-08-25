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


#ifndef RAJA_pattern_nested_For_HPP
#define RAJA_pattern_nested_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/policy/simd/policy.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that implements a single loop.
 *
 *
 */
template <camp::idx_t ArgumentId,
          typename ExecPolicy = camp::nil,
          typename... EnclosedStmts>
struct For : public internal::ForList,
             public internal::ForTraitBase<ArgumentId, ExecPolicy>,
             public internal::Statement<ExecPolicy, EnclosedStmts...> {

  // TODO: add static_assert for valid policy in Pol
  using execution_policy_t = ExecPolicy;
};

}  // end namespace statement

namespace internal
{


template <camp::idx_t ArgumentId, typename Data, typename... EnclosedStmts>
struct ForWrapper : public GenericWrapper<Data, EnclosedStmts...> {

  using Base = GenericWrapper<Data, EnclosedStmts...>;
  using Base::Base;
  using privatizer = NestedPrivatizer<ForWrapper>;

  template <typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
    Base::data.template assign_offset<ArgumentId>(i);
    Base::exec();
  }
};


/*!
 *Helper structs to check if we have a lambda
 *
 */
template<class T>
struct TypeIsLambda
{  
  static const bool value = false;
};

template <camp::idx_t BodyIdx>
struct TypeIsLambda<RAJA::statement::Lambda<BodyIdx> >
{
  static const bool value = true;
};

/*
 *
 */
template <class... States>
struct all {
};

template <>
struct all<>{
  static const bool value = true;
};

template<class State, class... States>
struct all<State, States...> : all<States...>
{
  static const bool value = TypeIsLambda<camp::decay<State>>::value;  
  static_assert((value && all<States ...>::value), "Lambdas are only supported post RAJA::simd_exec");
};


/*!
 * RAJA::kernel forall_impl executor specialization.
 * Assumptions: RAJA::simd_exec is the inner most policy, 
 * only one lambda is used, no reductions are done within the lambda.
 *
 */
template <camp::idx_t ArgumentId, typename... EnclosedStmts>
struct StatementExecutor<
    statement::For<ArgumentId, RAJA::simd_exec, EnclosedStmts...>> {

  template <camp::idx_t LoopIndex,
            camp::idx_t... OffsetIdx,
            camp::idx_t... ParamIdx,
            typename Data,
            typename Offs>
  static RAJA_INLINE void invoke_lambda_special(
      camp::idx_seq<OffsetIdx...> const &,
      camp::idx_seq<ParamIdx...> const &,
      Data &data,
      Offs const &offset_tuple)
  {
    camp::get<LoopIndex>(
        data.bodies)((camp::get<OffsetIdx>(data.segment_tuple)
                          .begin()[camp::get<OffsetIdx>(offset_tuple)])...,
                     camp::get<ParamIdx>(data.param_tuple)...);
  }

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    
    //assert we are only passing in Lambda objects
    all<EnclosedStmts...>::value;

    auto iter = get<ArgumentId>(data.segment_tuple);
    auto begin = std::begin(iter);
    auto end = std::end(iter);
    auto distance = std::distance(begin, end);
    RAJA_SIMD
    for (decltype(distance) i = 0; i < distance; ++i) {
      auto offsets = data.offset_tuple;
      auto param = data.param_tuple;
      get<ArgumentId>(offsets) = i;
      invoke_lambda_special<0>(camp::idx_seq_from_t<decltype(offsets)>{},
                               camp::idx_seq_from_t<decltype(param)>{},
                               data,
                               offsets);
    }
  }
};
    
/*!
 * A generic RAJA::kernel forall_impl executor
 * 
 *
 */
template <camp::idx_t ArgumentId,
          typename ExecPolicy,
          typename... EnclosedStmts>
struct StatementExecutor<statement::
                             For<ArgumentId, ExecPolicy, EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    // Create a wrapper, just in case forall_impl needs to thread_privatize
    ForWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);

    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    forall_impl(ExecPolicy{}, TypedRangeSegment<len_t>(0, len), for_wrapper);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_nested_HPP */
