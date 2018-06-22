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

#ifndef RAJA_policy_vec_forall_HPP
#define RAJA_policy_vec_forall_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/vec/policy.hpp"

namespace RAJA
{

namespace policy
{
namespace vec
{


template <typename VecType, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const vec_exec<VecType> &,
                             Iterable &&iter,
                             Func &&loop_body)
{
  static constexpr size_t element_width = VecType::num_total_elements;

  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  using distance_t = decltype(distance);

  using IdxType = decltype(*begin);

  using scalar_type = typename VecType::scalar_type;
  using ScalarType = RAJA::vec::Vector<scalar_type, 1, 1>;

  // vector-width loop
  distance_t vec_remainder = distance % element_width;
  distance_t vec_distance = distance - vec_remainder;

  using VecIdx = RAJA::vec::VectorIndex<IdxType, VecType>;
  for (distance_t i = 0; i < vec_distance; i+= element_width) {
    loop_body(VecIdx{*(begin + i)});
  }

  // postamble scalar-width loop
  using ScalarIdx = RAJA::vec::VectorIndex<IdxType, ScalarType>;
  for (distance_t i = vec_distance; i < distance; ++i) {
    loop_body(ScalarIdx{*(begin + i)});
  }

}

}  // closing brace for vec namespace

}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
