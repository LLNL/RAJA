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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_Statement_HPP
#define RAJA_pattern_kernel_internal_Statement_HPP

#include "RAJA/pattern/kernel/internal/StatementList.hpp"
#include <type_traits>
#include <camp/camp.hpp>

namespace RAJA
{
namespace internal
{


template<typename ExecPolicy, typename... EnclosedStmts>
struct Statement
{
  static_assert(std::is_same<ExecPolicy, camp::nil>::value ||
                    sizeof...(EnclosedStmts) > 0,
                "Executable statement with no enclosed statements, this is "
                "almost certainly a bug");
  Statement() = delete;

  using enclosed_statements_t = StatementList<EnclosedStmts...>;
  using execution_policy_t    = ExecPolicy;
};


template<typename Policy, typename Types>
struct StatementExecutor;


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_HPP */
