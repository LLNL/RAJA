//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Divide_HPP__
#define __TEST_TESNOR_REGISTER_Divide_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void DivideImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0)+1.0;
    B[i] = (element_t)(NO_OPT_RAND*1000.0)+1.0;
    x.set(A[i], i);
    y.set(B[i], i);
  }

  // operator /
  register_t op_div = x/y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_div.get(i), A[i] / B[i]);
  }

  // operator /=
  register_t op_diveq = x;
  op_diveq /= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_diveq.get(i), A[i] / B[i]);
  }

  // function divide
  register_t func_div = x.divide(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_div.get(i), A[i] / B[i]);
  }


  // operator / scalar
  register_t op_div_s1 = x / element_t(2);
  register_t op_div_s2 = element_t(2) / x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_div_s1.get(i), A[i] / element_t(2));
    ASSERT_SCALAR_EQ(op_div_s2.get(i), element_t(2) / A[i]);
  }

  // operator /= scalar
  register_t op_diveq_s = x;
  op_diveq_s /= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_diveq_s.get(i), A[i] / element_t(2));
  }
}



TYPED_TEST_P(TestTensorRegister, Divide)
{
  DivideImpl<TypeParam>();
}


#endif
