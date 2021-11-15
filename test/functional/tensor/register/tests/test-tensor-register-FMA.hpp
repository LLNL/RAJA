//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_REGISTER_FMA_HPP__
#define __TEST_TESNOR_REGISTER_FMA_HPP__

#include<RAJA/RAJA.hpp>

template <typename REGISTER_TYPE>
void FMAImpl()
{
  using register_t = REGISTER_TYPE;
  using element_t = typename register_t::element_type;
  using policy_t = typename register_t::register_policy;

  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem], C[num_elem], expected[num_elem];
  register_t x, y, z, result;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
    y.set(B[i], i);
    z.set(C[i], i);
    expected[i] = A[i]*B[i]+C[i];
  }

  result = x.multiply_add(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(result.get(i), expected[i]);
  }
}



TYPED_TEST_P(TestTensorRegister, FMA)
{
  FMAImpl<TypeParam>();
}


#endif
