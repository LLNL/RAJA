/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::kernel
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

#ifndef RAJA_pattern_kernel_HPP
#define RAJA_pattern_kernel_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/chai_support.hpp"

namespace RAJA
{

/*!
 * A RAJA::kernel execution policy.
 *
 * This is just a list of RAJA::kernel statements.
 */
template <typename... Stmts>
using KernelPolicy = internal::StatementList<Stmts...>;


///
/// Template list of argument indices
///
template <camp::idx_t... ArgumentId>
using ArgList = camp::idx_seq<ArgumentId...>;


template <typename T>
struct IterableWrapperTuple;

template <typename... Ts>
struct IterableWrapperTuple<camp::tuple<Ts...>> {

  using type =
      camp::tuple<RAJA::impl::Span<typename camp::decay<Ts>::iterator,
                                   typename camp::decay<Ts>::IndexType>...>;
};


namespace internal
{
template <class Tuple, camp::idx_t... I>
RAJA_INLINE constexpr auto make_wrapped_tuple_impl(Tuple &&t,
                                                   camp::idx_seq<I...>)
    -> camp::tuple<RAJA::impl::Span<
        typename camp::decay<
            camp::tuple_element_t<I, camp::decay<Tuple>>>::iterator,
        typename camp::decay<
            camp::tuple_element_t<I, camp::decay<Tuple>>>::IndexType>...>
{
  return camp::make_tuple(
      RAJA::impl::Span<
          typename camp::decay<
              camp::tuple_element_t<I, camp::decay<Tuple>>>::iterator,
          typename camp::decay<camp::tuple_element_t<I, camp::decay<Tuple>>>::
              IndexType>{camp::get<I>(std::forward<Tuple>(t)).begin(),
                         camp::get<I>(std::forward<Tuple>(t)).end()}...);
}
}  // namespace internal

template <class Tuple>
RAJA_INLINE constexpr auto make_wrapped_tuple(Tuple &&t)
    -> decltype(internal::make_wrapped_tuple_impl(
        std::forward<Tuple>(t),
        camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple>>::value>{}))
{
  return internal::make_wrapped_tuple_impl(
      std::forward<Tuple>(t),
      camp::make_idx_seq_t<camp::tuple_size<camp::decay<Tuple>>::value>{});
}


template <typename PolicyType,
          typename SegmentTuple,
          typename ParamTuple,
          typename... Bodies>
RAJA_INLINE void kernel_param(SegmentTuple &&segments,
                              ParamTuple &&params,
                              Bodies &&... bodies)
{

  detail::setChaiExecutionSpace<PolicyType>();

  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple
  // TODO: add assert that all Lambda<i> match supplied loop bodies

  using segment_tuple_t =
      typename IterableWrapperTuple<camp::decay<SegmentTuple>>::type;


  using param_tuple_t = camp::decay<ParamTuple>;

  using loop_data_t = internal::LoopData<PolicyType,
                                         segment_tuple_t,
                                         param_tuple_t,
                                         camp::decay<Bodies>...>;


  // Create the LoopData object, which contains our policy object,
  // our segments, loop bodies, and the tuple of loop indices
  // it is passed through all of the kernel mechanics by-referenece,
  // and only copied to provide thread-private instances.
  loop_data_t loop_data(make_wrapped_tuple(
                            std::forward<SegmentTuple>(segments)),
                        std::forward<ParamTuple>(params),
                        std::forward<Bodies>(bodies)...);


  // Execute!
  RAJA_FORCEINLINE_RECURSIVE
  internal::execute_statement_list<PolicyType>(loop_data);


  detail::clearChaiExecutionSpace();
}

template <typename PolicyType, typename SegmentTuple, typename... Bodies>
RAJA_INLINE void kernel(SegmentTuple &&segments, Bodies &&... bodies)
{
  RAJA::kernel_param<PolicyType>(std::forward<SegmentTuple>(segments),
                                 RAJA::make_tuple(),
                                 std::forward<Bodies>(bodies)...);
}


}  // end namespace RAJA


#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/Conditional.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/ForICount.hpp"
#include "RAJA/pattern/kernel/Hyperplane.hpp"
#include "RAJA/pattern/kernel/InitLocalMem.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"
#include "RAJA/pattern/kernel/Reduce.hpp"
#include "RAJA/pattern/kernel/Tile.hpp"
#include "RAJA/pattern/kernel/TileTCount.hpp"


#endif /* RAJA_pattern_kernel_HPP */
