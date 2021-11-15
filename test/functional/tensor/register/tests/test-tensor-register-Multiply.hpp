//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Multiply_HPP__
#define __TEST_TESNOR_REGISTER_Multiply_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void MultiplyImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
    y.set(B[i], i);
  }

  // operator *
  register_t op_mul = x*y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_mul.get(i), (A[i] * B[i]));
  }

  // operator *=
  register_t op_muleq = x;
  op_muleq *= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_muleq.get(i), A[i] * B[i]);
  }

  // function multiply
  register_t func_mul = x.multiply(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_mul.get(i), A[i] * B[i]);
  }

  // operator * scalar
  register_t op_mul_s1 = x * element_t(2);
  register_t op_mul_s2 = element_t(2) * x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_mul_s1.get(i), A[i] * element_t(2));
    ASSERT_SCALAR_EQ(op_mul_s2.get(i), element_t(2) * A[i]);
  }

  // operator *= scalar
  register_t op_muleq_s = x;
  op_muleq_s *= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_muleq_s.get(i), A[i] * element_t(2));
  }
}



TYPED_TEST_P(TestTensorRegister, Multiply)
{
  MultiplyImpl<TypeParam>();
}


#endif
