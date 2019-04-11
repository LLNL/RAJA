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

namespace statement
{


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

template <camp::idx_t LoopIndex>
struct StatementExecutor<statement::Lambda<LoopIndex>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    invoke_lambda<LoopIndex>(std::forward<Data>(data));
  }
};

//...extractor of arguments
template<typename Head, typename...Tail>
struct extractor
{};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::Seg<id>, Tail...>
{
  template<typename Data>
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)])
  {
    return camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)];
  }
};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::Param<id>, Tail...>
{
  template<typename Data>
  static auto extract_arg(Data &&data)->
    decltype(camp::get<id>(data.param_tuple))
  {
    return camp::get<id>(data.param_tuple);
  }
};

//Lambda with custom args
template <camp::idx_t LoopIndex,typename... Args>
struct StatementExecutor<statement::Lambda<LoopIndex, Args...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    const int size = sizeof...(Args); //Total number of arguments
    auto myTuple = camp::make_tuple(extractor<Args>::extract_arg(data) ...);

    qinvoke_lambda<LoopIndex>(std::forward<Data>(data), myTuple,camp::make_idx_seq_t<size>{});
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
