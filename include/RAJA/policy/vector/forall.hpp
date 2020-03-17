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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_vector_HPP
#define RAJA_forall_vector_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/vector/policy.hpp"

namespace RAJA
{
namespace policy
{
namespace vector
{


template <typename VectorType, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const vector_exec<VectorType>&,
                             Iterable &&iter,
                             Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  using diff_t = decltype(distance);

  using value_type = typename Iterable::value_type;
  using vector_type = VectorType;
  using vector_index_type = VectorIndex<value_type, vector_type>;

  diff_t distance_simd = distance - (distance%vector_type::s_num_elem);
  diff_t distance_remainder = distance - distance_simd;

  // Streaming loop for complete vector widths
  for (diff_t i = 0; i < distance_simd; i+=vector_type::s_num_elem) {
    loop_body(vector_index_type(*(begin + i)));
  }

  // Postamble for reamining elements
  if(distance_remainder > 0){
    loop_body(vector_index_type(*(begin + distance_simd), distance_remainder));
  }

}


}  // namespace vector

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
