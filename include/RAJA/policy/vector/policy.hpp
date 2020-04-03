/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA simd policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_vector_policy_HPP
#define RAJA_policy_vector_policy_HPP

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/config.hpp"
#include "RAJA/pattern/vector.hpp"
#include "RAJA/policy/vector/register.hpp"


//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
namespace RAJA
{
namespace policy
{
namespace vector
{

template<typename TENSOR_TYPE, camp::idx_t DIM>
struct tensor_exec : make_policy_pattern_launch_platform_t<Policy::sequential,
                                                         Pattern::forall,
                                                         Launch::undefined,
                                                         Platform::host> {
  using tensor_type = TENSOR_TYPE;

  static constexpr camp::idx_t s_tensor_dim = DIM;
};



}  // end of namespace vector

}  // end of namespace policy

template<typename VECTOR_TYPE>
using vector_exec = policy::vector::tensor_exec<VECTOR_TYPE, 0>;

template<typename MATRIX_TYPE>
using matrix_row_exec = policy::vector::tensor_exec<MATRIX_TYPE, 0>;

template<typename MATRIX_TYPE>
using matrix_col_exec = policy::vector::tensor_exec<MATRIX_TYPE, 1>;





}  // end of namespace RAJA

#endif
