//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Add_HPP__
#define __TEST_TESNOR_REGISTER_Add_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void AddImpl()
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

   // operator +
   register_t op_add = x+y;
   for(size_t i = 0;i < num_elem; ++ i){
     ASSERT_SCALAR_EQ(op_add.get(i), A[i] + B[i]);
   }

   // operator +=
   register_t op_pluseq = x;
   op_pluseq += y;
   for(size_t i = 0;i < num_elem; ++ i){
     ASSERT_SCALAR_EQ(op_pluseq.get(i), A[i] + B[i]);
   }

   // function add
   register_t func_add = x.add(y);
   for(size_t i = 0;i < num_elem; ++ i){
     ASSERT_SCALAR_EQ(func_add.get(i), A[i] + B[i]);
   }

   // operator + scalar
   register_t op_add_s1 = x + element_t(1);
   register_t op_add_s2 = element_t(1) + x;
   for(size_t i = 0;i < num_elem; ++ i){
     ASSERT_SCALAR_EQ(op_add_s1.get(i), A[i] + element_t(1));
     ASSERT_SCALAR_EQ(op_add_s2.get(i), element_t(1) + A[i]);
   }

   // operator += scalar
   register_t op_pluseq_s = x;
   op_pluseq_s += element_t(1);
   for(size_t i = 0;i < num_elem; ++ i){
     ASSERT_SCALAR_EQ(op_pluseq_s.get(i), A[i] + element_t(1));
   }

}



TYPED_TEST_P(TestTensorRegister, Add)
{
  AddImpl<TypeParam>();
}


#endif
