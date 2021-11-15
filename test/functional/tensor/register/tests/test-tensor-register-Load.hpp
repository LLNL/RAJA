//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Load_HPP__
#define __TEST_TESNOR_REGISTER_Load_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void LoadImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*2];
  for(size_t i = 0;i < num_elem*2; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
  }


  // load stride-1 from pointer
  register_t x;
  x.load_packed(A);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
  }


  // load n stride-1 from pointer
  if(num_elem > 1){
    x.load_packed_n(A, num_elem-1);

    // check first n-1 values
    for(size_t i = 0;i+1 < num_elem; ++ i){
      ASSERT_SCALAR_EQ(x.get(i), A[i]);
    }

    // last value should be cleared to zero
    ASSERT_SCALAR_EQ(x.get(num_elem-1), 0);
  }

  // load stride-2 from pointer
  register_t y;
  y.load_strided(A, 2);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(y.get(i), A[i*2]);
  }

  // load n stride-2 from pointer
  if(num_elem > 1){
    y.load_strided_n(A, 2, num_elem-1);

    // check first n-1 values
    for(size_t i = 0;i+1 < num_elem; ++ i){
      ASSERT_SCALAR_EQ(y.get(i), A[i*2]);
    }

    // last value should be cleared to zero
    ASSERT_SCALAR_EQ(y.get(num_elem-1), 0);
  }
}



TYPED_TEST_P(TestTensorRegister, Load)
{
  LoadImpl<TypeParam>();
}


#endif
