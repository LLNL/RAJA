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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
RAJA_INLINE void forall_impl(const loop_exec &, Iterable &&iter, Func &&body)
{
  RAJA_EXTRACT_BED_IT(iter);

  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    body(*(begin_it + i));
  }
}

}  // closing brace for loop namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
