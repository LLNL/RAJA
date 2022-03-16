//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_MATRIX_ET_Negate_HPP__
#define __TEST_TESNOR_MATRIX_ET_Negate_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_NegateImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  //
  // Allocate Row-Major Data
  //

  // alloc input0

  std::vector<element_t> input0_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> input0_h(input0_vec.data(), N, N);

  element_t *input0_ptr = tensor_malloc<policy_t>(input0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> input0_d(input0_ptr,  N, N);



  // alloc output0

  std::vector<element_t> output0_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> output0_h(output0_vec.data(),  N, N);

  element_t *output0_ptr = tensor_malloc<policy_t>(output0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output0_d(output0_ptr,  N, N);



  // Fill input0 and output0
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      input0_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(input0_ptr, input0_vec);


  //
  // Do Operation: negation
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    output0_d(rows, cols) = -input0_d(rows, cols);

  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(output0_h(i,j), -input0_h(i,j));
    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(input0_ptr);
  tensor_free<policy_t>(output0_ptr);

}



TYPED_TEST_P(TestTensorMatrix, ET_Negate)
{
  ET_NegateImpl<TypeParam>();
}


#endif
