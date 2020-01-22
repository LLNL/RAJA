/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_Statement_HPP
#define RAJA_pattern_kernel_internal_Statement_HPP

#include "RAJA/pattern/kernel/internal/StatementList.hpp"

namespace RAJA
{
namespace internal
{



template <typename ExecPolicy, typename... EnclosedStmts>
struct Statement {
  Statement() = delete;

  using enclosed_statements_t = StatementList<EnclosedStmts...>;
  using execution_policy_t = ExecPolicy;
};




template <typename Policy, typename Types>
struct StatementExecutor;



}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_HPP */
