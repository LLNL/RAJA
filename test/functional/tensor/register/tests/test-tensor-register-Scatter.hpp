//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Scatter_HPP__
#define __TEST_TESNOR_REGISTER_Scatter_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void ScatterImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  using int_register_t = typename register_t::int_vector_type;


  static constexpr camp::idx_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*num_elem];
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
    A[i] = 0;
  }

  // create an index vector to point at sub elements of A
  int_register_t idx;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
    idx.set(j*j, i);
  }

  // Create a vector of values
  register_t x;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    x.set(i+1, i);
  }

  // Scatter the values of x into A[] using idx as the offsets
  x.scatter(&A[0], idx);

  // check
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
    // if the index i is in idx, check that A contains the right value
    for(camp::idx_t j = 0;j < num_elem;++ j){
      if(idx.get(j) == i){
        // check
        ASSERT_SCALAR_EQ(A[i], element_t(j+1));
        // and set to zero (for the next assert, and to clear for next test)
        A[i] = 0;
      }
    }
    // otherwise A should contain zero
    ASSERT_SCALAR_EQ(A[i], element_t(0));
  }


  // Scatter all but one of the values of x into A[] using idx as the offsets
  x.scatter_n(&A[0], idx, num_elem-1);

  // check
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
    // if the index i is in idx, check that A contains the right value
    for(camp::idx_t j = 0;j < num_elem-1;++ j){
      if(idx.get(j) == i){
        // check
        ASSERT_SCALAR_EQ(A[i], element_t(j+1));
        // and set to zero (for the next assert, and to clear for next test)
        A[i] = 0;
      }
    }
    // otherwise A should contain zero
    ASSERT_SCALAR_EQ(A[i], element_t(0));
  }


}



TYPED_TEST_P(TestTensorRegister, Scatter)
{
  ScatterImpl<TypeParam>();
}


#endif
