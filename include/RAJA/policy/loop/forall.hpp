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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#include "camp/camp.hpp"

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


//
//////////////////////////////////////////////////////////////////////
//
// The following provide unrolled loop execution.
//
// The unroll_loop_exec<unroll_list...> policy provides loop unrolling
// for sizes indicated in the template parameters.
//
// Parameters should be provided in DECREASING order
//
//////////////////////////////////////////////////////////////////////
//
namespace detail {

  // Termination case: no more unrolling, so treat as normal stride-1 loop
  template <typename BeginIter, typename Func>
  RAJA_INLINE void forall_unroll(const unroll_loop_exec<> &, BeginIter &&begin_it, size_t remaining, Func &&body)
  {
    // Iterate over remaining one-by-one
    for (size_t i = 0; i < remaining; ++i) {
      body(*(begin_it + i));
    }
  }

  // Tries to unroll the loop by "unroll" until we run out of enough iterations
  // remainder is passed to next smallest size (or 1 as a termination case)
  template <typename BeginIter, typename Func, size_t unroll, size_t ... unroll_next>
  RAJA_INLINE void forall_unroll(const unroll_loop_exec<unroll, unroll_next...> &, BeginIter &&begin_it, size_t remaining, Func &&body)
  {
    // Iterate over "unroll" lengthed chunks
    if(remaining >= unroll){
      for (size_t i = 0; i+unroll <= remaining; i += unroll) {

        // Fixed length loop.  Should unroll, but we could possibly use
        // a compiler specific pragma to force it?!?
        for(size_t j = 0;j < unroll; ++j) {
          body(*(begin_it + j));
        }
        begin_it += unroll;
      }
    }

    // If we have any iterations remaining, pass them on to the next smaller
    // sized unroll
    size_t remainder = remaining % unroll;
    if(remainder > 0){
      using next_pol = unroll_loop_exec<unroll_next ...>;

      forall_unroll(next_pol{}, begin_it, remainder, std::forward<Func>(body));
    }
  }





} // namespace detail



template <typename Iterable, typename Func, size_t ... unroll_list>
RAJA_INLINE void forall_impl(const unroll_loop_exec<unroll_list...> &pol, Iterable &&iter, Func &&body)
{
  RAJA_EXTRACT_BED_IT(iter);

  detail::forall_unroll(pol, begin_it, (size_t)distance_it, std::forward<Func>(body));
}


}  // closing brace for loop namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
