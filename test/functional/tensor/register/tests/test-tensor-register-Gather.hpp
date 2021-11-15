//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Gather_HPP__
#define __TEST_TESNOR_REGISTER_Gather_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void GatherImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  using int_register_t = typename register_t::int_vector_type;

  static constexpr camp::idx_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*num_elem];
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
    A[i] = 3*i+13;
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
  }

  // create an index vector to point at sub elements of A
  int_register_t idx;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
    idx.set(j*j, i);
//    printf("idx[%d]=%d\n", (int)i, (int)(j*j));
  }

  // Gather elements from A into a register using the idx offsets
  register_t x;
  x.gather(&A[0], idx);

  // check
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
//    printf("i=%d, j=%d, A[%d]=%d, x.get(i)=%d\n",
//        (int)i, (int)j, (int)(j*j), (int)A[j*j], (int)x.get(i));
    ASSERT_SCALAR_EQ(A[j*j], x.get(i));
  }


  // Gather all but one elements from A into a register using the idx offsets
  register_t y;
  y.gather_n(&A[0], idx, num_elem-1);

  // check
  for(camp::idx_t i = 0;i < num_elem-1;++ i){
    int j = num_elem-1-i;
    ASSERT_SCALAR_EQ(A[j*j], y.get(i));
  }
  ASSERT_SCALAR_EQ(0, y.get(num_elem-1));

}



TYPED_TEST_P(TestTensorRegister, Gather)
{
  GatherImpl<TypeParam>();
}


#endif
