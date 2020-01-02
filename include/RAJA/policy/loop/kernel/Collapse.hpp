/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel loop collapse executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_loop_kernel_Collapse_HPP
#define RAJA_policy_loop_kernel_Collapse_HPP

#include <RAJA/pattern/kernel.hpp>

namespace RAJA
{

namespace internal
{


//
// Termination case for seq_exec collapsed loops
//
template <typename... EnclosedStmts>
struct StatementExecutor<
    statement::Collapse<loop_exec, ArgList<>, EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &data)
  {
    // termination case: no more loops, just execute enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>>(data);
  }
};


//
// Executor that handles collapsing of an arbitrarily deep set of seq_exec
// loops
//
template <camp::idx_t Arg0, camp::idx_t... ArgRest, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<loop_exec,
                                             ArgList<Arg0, ArgRest...>,
                                             EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &data)
  {
    // compute next-most inner loop Executor
    using next_loop_t = StatementExecutor<
        statement::Collapse<loop_exec, ArgList<ArgRest...>, EnclosedStmts...>>;

    auto len0 = segment_length<Arg0>(data);

    for (auto i0 = 0; i0 < len0; ++i0) {
      data.template assign_offset<Arg0>(i0);

      next_loop_t::exec(data);
    }
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
