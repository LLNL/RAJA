/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for loop execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_loop_HPP
#define RAJA_forall_loop_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

using RAJA::concepts::enable_if;

namespace RAJA
{
namespace policy
{
namespace loop
{

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//


template <typename Iterable, typename Func>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(RAJA::resources::Host & host_res,
                                                    const loop_exec &,
                                                    Iterable &&iter,
                                                    Func &&body)
{
  RAJA_EXTRACT_BED_IT(iter);

  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    body(*(begin_it + i));
  }
  return RAJA::resources::EventProxy<resources::Host>(&host_res);
}

}  // namespace loop

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
