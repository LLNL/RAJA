/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sequential_WorkGroup_HPP
#define RAJA_sequential_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

#include "RAJA/policy/loop/WorkGroup.hpp"
#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{

namespace detail
{

/*!
* Populate and return a Vtable object
*/
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(seq_work const&)
{
  return get_Vtable<T, CallArgs...>(loop_work{});
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
