/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 *          Note: Reduction operations should not be used with simd
 *          policies. Limited support.
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_simd_HPP
#define RAJA_forall_simd_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/simd/policy.hpp"

namespace RAJA
{
namespace policy
{
namespace simd
{


template <typename Iterable, typename Func>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(RAJA::resources::Host &host_res,
                                                               const simd_exec &,
                                                               Iterable &&iter,
                                                               Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  RAJA_SIMD
  for (decltype(distance) i = 0; i < distance; ++i) {
    loop_body(*(begin + i));
  }

  return RAJA::resources::EventProxy<resources::Host>(&host_res);
}

}  // namespace simd

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
