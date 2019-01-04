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
          typename... EnclosedStmts>
struct StatementExecutor<
    statement::Reduce<seq_reduce, ReduceOperator, ParamId, EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    // since a sequential reduction is a NOP, and the single thread always
    // has the reduced value, this is just a passthrough to the enclosed
    // statements
    execute_statement_list<camp::list<EnclosedStmts...>>(data);
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_policy_sequential_kernel_Reduce_HPP */
