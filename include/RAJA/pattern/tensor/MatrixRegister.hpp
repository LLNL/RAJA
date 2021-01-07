/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_MatrixRegister_HPP
#define RAJA_pattern_tensor_MatrixRegister_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"

namespace RAJA
{

  template<typename T, typename LAYOUT, typename REGISTER_POLICY = RAJA::default_register>
  using MatrixRegister =
      TensorRegister<REGISTER_POLICY,
                     T,
                     LAYOUT,
                     camp::idx_seq<RegisterTraits<REGISTER_POLICY,T>::s_num_elem,
                                   RegisterTraits<REGISTER_POLICY,T>::s_num_elem>,
                     camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY,T>::s_num_elem>>;




}  // namespace RAJA



#endif
