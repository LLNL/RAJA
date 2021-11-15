//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Min_HPP__
#define __TEST_TESNOR_REGISTER_Min_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void MinImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

   for(int iter = 0;iter < 100;++ iter){
     element_t A[num_elem], B[num_elem];
     register_t x, y;

     for(size_t i = 0;i < num_elem; ++ i){
       A[i] = (element_t)(NO_OPT_RAND*1000.0);
       B[i] = (element_t)(NO_OPT_RAND*1000.0);
       x.set(A[i], i);
       y.set(B[i], i);
     }

     // Check vector reduction
     element_t expected = A[0];
     for(size_t i = 1;i < num_elem;++ i){
       expected = expected < A[i] ? expected : A[i];
     }

 //    printf("X=%s", x.to_string().c_str());


     ASSERT_SCALAR_EQ(x.min(), expected);

     // Check element-wise
     register_t z = x.vmin(y);
     for(size_t i = 1;i < num_elem;++ i){
       ASSERT_SCALAR_EQ(z.get(i), std::min<element_t>(A[i], B[i]));
     }

   }
}



TYPED_TEST_P(TestTensorRegister, Min)
{
  MinImpl<TypeParam>();
}


#endif
