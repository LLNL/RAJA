//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_Store_HPP__
#define __TEST_TESNOR_REGISTER_Store_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void StoreImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
  }

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
  }

  // test copy construction
  register_t cc(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(cc.get(i), A[i]);
  }

  // test explicit copy
  register_t ce(0);
  ce.copy(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ce.get(i), A[i]);
  }

  // test assignment
  register_t ca(0);
  ca = cc;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ca.get(i), A[i]);
  }

  // test scalar construction (broadcast)
  register_t bc((element_t)5);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(bc.get(i), 5.0);
  }

  // test scalar assignment (broadcast)
  register_t ba((element_t)0);
  ba = (element_t)13.0;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ba.get(i), 13.0);
  }

  // test explicit broadcast
  register_t be((element_t)0);
  be.broadcast((element_t)13.0);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(be.get(i), 13.0);
  }
}



TYPED_TEST_P(TestTensorRegister, Store)
{
  StoreImpl<TypeParam>();
}


#endif
