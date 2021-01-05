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

#ifndef RAJA_policy_tensor_vectorregister_HPP
#define RAJA_policy_tensor_vectorregister_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/internal/VectorRegisterBase.hpp"
#include "RAJA/policy/tensor/arch.hpp"

namespace RAJA
{

  // Convenience to describe VectorTensors
  template<typename T, typename REGISTER_POLICY = default_register>
  using VectorRegister = TensorRegister<REGISTER_POLICY,
                                        T,
                                        VectorLayout,
                                        camp::idx_seq<RegisterTraits<REGISTER_POLICY,T>::s_num_elem>,
                                        camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY,T>::s_num_elem>,
                                        0>;

} // namespace RAJA



#endif
