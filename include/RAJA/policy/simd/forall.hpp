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



template <typename VectorType, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const simd_vector_exec<VectorType> &,
                             Iterable &&iter,
                             Func &&loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);

  auto distance_simd = distance - (distance%VectorType::s_num_elem);
  auto distance_remainder = distance - distance_simd;

  using index_type = camp::decay<decltype(*begin)>;
  using simd_index_type = VectorIndex<index_type, VectorType>;

  // Streaming loop for complete vector widths
  for (decltype(distance) i = 0; i < distance_simd; i+=VectorType::s_num_elem) {
    loop_body(simd_index_type(*(begin + i), VectorType::s_num_elem));
  }

  // Postamble for reamining elements
  if(distance_remainder > 0){
    loop_body(simd_index_type(*(begin + distance_simd), distance_remainder));
  }

}


}  // namespace simd

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
