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


template <typename TENSOR_TYPE, camp::idx_t DIM, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const tensor_exec<TENSOR_TYPE, DIM>&,
                             Iterable &&iter,
                             Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
  using diff_t = decltype(distance);

  using value_type = typename Iterable::value_type;
  using tensor_type = TENSOR_TYPE;
  using tensor_index_type = TensorIndex<value_type, tensor_type, DIM>;

  diff_t distance_simd = distance - (distance%tensor_type::s_dim_elem(DIM));
  diff_t distance_remainder = distance - distance_simd;

  // Streaming loop for complete vector widths
  for (diff_t i = 0; i < distance_simd; i+=tensor_type::s_dim_elem(DIM)) {
    loop_body(tensor_index_type(*(begin + i)));
  }

  // Postamble for reamining elements
  if(distance_remainder > 0){
    loop_body(tensor_index_type(*(begin + distance_simd), distance_remainder));
  }

}


}  // namespace vector

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
