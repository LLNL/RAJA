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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_tensor_scalarregister_HPP
#define RAJA_policy_tensor_scalarregister_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/VectorRegister.hpp"
#include "RAJA/policy/tensor/arch.hpp"


namespace RAJA
{
namespace expt
{

  // Convenience to describe ScalarTensors
  template<typename T>
  using ScalarRegister = TensorRegister<scalar_register,
                                        T,
                                        ScalarLayout,
                                        camp::idx_seq<>>;


} // namespace expt
} // namespace RAJA


#endif
