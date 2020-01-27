/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel lambda executor.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_Lambda_HPP
#define RAJA_pattern_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{

namespace internal
{
struct seg_t
{};

struct param_t
{};

struct offset_t
{};

template<typename T, camp::idx_t>
struct LambdaArg
{
};

}

namespace statement
{


template<camp::idx_t ... args>
using Segs = camp::list<internal::LambdaArg<internal::seg_t, args>...>;

template<camp::idx_t ... args>
using Offsets = camp::list<internal::LambdaArg<internal::offset_t, args>...>;

template<camp::idx_t ... args>
using Params = camp::list<internal::LambdaArg<internal::param_t, args>...>;



/*!
 * A RAJA::kernel statement that invokes a lambda function.
 *
 * The lambda is specified by its index in the sequence of lambda arguments
 * to a RAJA::kernel method.
 *
 * for example:
 * RAJA::kernel<exec_pol>(make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template <camp::idx_t BodyIdx, typename... Args >
struct Lambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};

}  // end namespace statement

namespace internal
{


//Extracts arguments from segments, and parameters
template<typename T>
struct LambdaArgExtractor;

template<camp::idx_t id>
struct LambdaArgExtractor<LambdaArg<offset_t, id>>
{

  template<typename Data>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.offset_tuple))
  {
    return camp::get<id>(data.offset_tuple);
  }

};

template<camp::idx_t id>
struct LambdaArgExtractor<LambdaArg<seg_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)])
  {
    return camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)];
  }
};

template<camp::idx_t id>
struct LambdaArgExtractor<LambdaArg<param_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  static auto extract_arg(Data &&data)->
    typename std::add_lvalue_reference<camp::tuple_element_t<id,typename camp::decay<Data>::param_tuple_t>>::type
  {
    return camp::get<id>(data.param_tuple);
  }
};


RAJA_SUPPRESS_HD_WARN
template <camp::idx_t LoopIndex,
          camp::idx_t... OffsetIdx,
          camp::idx_t... ParamIdx,
          typename Data>
RAJA_HOST_DEVICE RAJA_INLINE void invoke_lambda_expanded(
    camp::idx_seq<OffsetIdx...> const &,
    camp::idx_seq<ParamIdx...> const &,
    Data &&data)
{
  camp::get<LoopIndex>(data.bodies)
    ((camp::get<OffsetIdx>(data.segment_tuple).begin()[camp::get<OffsetIdx>(data.offset_tuple)])...,
     camp::get<ParamIdx>(data.param_tuple)...);
}


template <camp::idx_t LoopIndex, typename Data>
RAJA_INLINE RAJA_HOST_DEVICE void invoke_lambda(Data &&data)
{
  using Data_t = camp::decay<Data>;
  using offset_tuple_t = typename Data_t::offset_tuple_t;
  using param_tuple_t = typename Data_t::param_tuple_t;

  invoke_lambda_expanded<LoopIndex>(
      camp::make_idx_seq_t<camp::tuple_size<offset_tuple_t>::value>{},
      camp::make_idx_seq_t<camp::tuple_size<param_tuple_t>::value>{},
      std::forward<Data>(data));
}

RAJA_SUPPRESS_HD_WARN
template<camp::idx_t LoopIndex, typename Data, typename... targLists>
RAJA_INLINE RAJA_HOST_DEVICE void invoke_custom_lambda(Data &&data,
                                                       camp::list<targLists...> const &)
{
  camp::get<LoopIndex>(data.bodies)(LambdaArgExtractor<targLists>::extract_arg(data)...);
}

//Helper to launch lambda with custom arguments
template <camp::idx_t LoopIndex, typename targList, typename Data>
RAJA_INLINE RAJA_HOST_DEVICE void invoke_lambda_with_args(Data &&data)
{

  invoke_custom_lambda<LoopIndex>(data,targList{});

}


template <camp::idx_t LambdaIndex, typename Types>
struct StatementExecutor<statement::Lambda<LambdaIndex>, Types> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    invoke_lambda<LambdaIndex>(std::forward<Data>(data));
  }
};


/*!
 * A RAJA::kernel statement that invokes a lambda function
 * with user specified arguments.
 */
template <camp::idx_t LambdaIndex,typename... Args, typename Types>
struct StatementExecutor<statement::Lambda<LambdaIndex, Args...>, Types> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    //Convert SegList, ParamList into Seg, Param types, and store in a list
    using targList = typename camp::flatten<camp::list<Args...>>::type;

    invoke_lambda_with_args<LambdaIndex, targList>(std::forward<Data>(data));
  }
};

}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
