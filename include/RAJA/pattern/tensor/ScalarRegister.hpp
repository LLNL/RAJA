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

#ifndef RAJA_policy_tensor_scalarregister_HPP
#define RAJA_policy_tensor_scalarregister_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/VectorRegister.hpp"
#include "RAJA/policy/tensor/arch.hpp"


namespace RAJA
{


  // Convenience to describe ScalarTensors
  template<typename T>
  using ScalarRegister = TensorRegister<scalar_register,
                                        T,
                                        ScalarLayout,
                                        camp::idx_seq<>,
                                        camp::idx_seq<>>;



  template<typename T>
  class TensorRegister<scalar_register, T, ScalarLayout, camp::idx_seq<>, camp::idx_seq<>> :
    public VectorRegister<T, scalar_register>
  {
    public:
      using self_type = TensorRegister<scalar_register, T, ScalarLayout, camp::idx_seq<>, camp::idx_seq<>>;
      using base_type = VectorRegister<T, scalar_register>;
      using register_policy = scalar_register;
      using element_type = T;
      using register_type = T;


      RAJA_HOST_DEVICE
      RAJA_INLINE
      TensorRegister(element_type const &v) :
        base_type(v){}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      operator T() const {
        return base_type::get(0);
      }

  };

} // namespace RAJA


#endif
