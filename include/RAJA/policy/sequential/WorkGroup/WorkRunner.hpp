/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkRunner class specializations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sequential_WorkGroup_WorkRunner_HPP
#define RAJA_sequential_WorkGroup_WorkRunner_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template <typename DISPATCH_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename... Args>
struct WorkRunner<RAJA::seq_work,
                  RAJA::ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...> : WorkRunnerForallOrdered<RAJA::seq_exec,
                                                     RAJA::seq_work,
                                                     RAJA::ordered,
                                                     DISPATCH_POLICY_T,
                                                     ALLOCATOR_T,
                                                     INDEX_T,
                                                     Args...>
{};

/*!
 * Runs work in a storage container in reverse order
 * and returns any per run resources
 */
template <typename DISPATCH_POLICY_T,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename... Args>
struct WorkRunner<RAJA::seq_work,
                  RAJA::reverse_ordered,
                  DISPATCH_POLICY_T,
                  ALLOCATOR_T,
                  INDEX_T,
                  Args...> : WorkRunnerForallReverse<RAJA::seq_exec,
                                                     RAJA::seq_work,
                                                     RAJA::reverse_ordered,
                                                     DISPATCH_POLICY_T,
                                                     ALLOCATOR_T,
                                                     INDEX_T,
                                                     Args...>
{};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
