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

namespace RAJA
{

  // Convenience to describe VectorTensors
  template<typename T, typename REGISTER_POLICY = default_register>
  using VectorRegister = TensorRegister<REGISTER_POLICY,
                                        T,
                                        VectorLayout,
                                        camp::idx_seq<RegisterTraits<REGISTER_POLICY,T>::s_num_elem> >;

} // namespace RAJA



#endif
