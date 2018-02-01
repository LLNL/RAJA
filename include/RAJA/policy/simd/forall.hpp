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

#ifndef RAJA_forall_simd_HPP
#define RAJA_forall_simd_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

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
RAJA_INLINE void forall_impl(const simd_exec &,
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
}

}  // closing brace for simd namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
