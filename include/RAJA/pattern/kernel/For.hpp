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


#ifndef RAJA_pattern_kernel_For_HPP
#define RAJA_pattern_kernel_For_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that implements a single loop.
 * Assigns the loop iterate to argument ArgumentId
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

/*!
 * A generic RAJA::kernel forall_impl loop wrapper for statement::For
 * Assigns the loop index to offset ArgumentId
 *
 */
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
 * A generic RAJA::kernel forall_impl executor for statement::For
 *
 *
 */
template <camp::idx_t ArgumentId,
          typename ExecPolicy,
          typename... EnclosedStmts>
struct StatementExecutor<
    statement::For<ArgumentId, ExecPolicy, EnclosedStmts...>> {


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
