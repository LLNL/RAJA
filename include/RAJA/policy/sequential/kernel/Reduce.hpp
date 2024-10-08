/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for sequential kernel loop collapse executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_sequential_kernel_Reduce_HPP
#define RAJA_policy_sequential_kernel_Reduce_HPP

#include "RAJA/pattern/kernel.hpp"

namespace RAJA
{

namespace internal
{

//
// Executor that handles reductions for
//
template <template <typename...> class ReduceOperator,
          typename ParamId,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<
    statement::Reduce<seq_reduce, ReduceOperator, ParamId, EnclosedStmts...>,
    Types>
{

  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    // since a sequential reduction is a NOP, and the single thread always
    // has the reduced value, this is just a passthrough to the enclosed
    // statements
    execute_statement_list<camp::list<EnclosedStmts...>, Types>(data);
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_policy_sequential_kernel_Reduce_HPP */
