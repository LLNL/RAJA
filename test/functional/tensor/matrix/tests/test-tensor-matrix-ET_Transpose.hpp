//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_MATRIX_ET_Transpose_HPP__
#define __TEST_TESNOR_MATRIX_ET_Transpose_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_TransposeImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using transpose_t = typename matrix_t::transpose_type;

//  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;
//  static constexpr camp::idx_t M = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  static constexpr camp::idx_t N = matrix_t::s_num_rows;
  static constexpr camp::idx_t M = matrix_t::s_num_columns;

  //
  // Allocate Row-Major Data
  //

  // alloc input0

  std::vector<element_t> input0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> input0_h(input0_vec.data(), N, M);

  element_t *input0_ptr = tensor_malloc<policy_t>(input0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> input0_d(input0_ptr,  N, M);



  // alloc output0

  std::vector<element_t> output0_vec(N*M);
  RAJA::View<element_t, RAJA::Layout<2>> output0_h(output0_vec.data(),  M, N);

  element_t *output0_ptr = tensor_malloc<policy_t>(output0_vec);
  RAJA::View<element_t, RAJA::Layout<2>> output0_d(output0_ptr,  M, N);



  // Fill input0 and output0
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < M; ++ j){
      input0_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(input0_ptr, input0_vec);


  //
  // Do Operation: transpose
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    auto rows_tr = RAJA::RowIndex<int, transpose_t>::all();
    auto cols_tr = RAJA::ColIndex<int, transpose_t>::all();

    output0_d(rows_tr, cols_tr) = input0_d(rows, cols).transpose();

  });

  tensor_copy_to_host<policy_t>(output0_vec, output0_ptr);



//  for(camp::idx_t i = 0;i < M; ++ i){
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%3d ", (int)output0_h(i,j));
//    }
//    printf("\n");
//  }



  //
  // Check results
  //
  for(camp::idx_t i = 0;i < M; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(output0_h(i,j), input0_h(j,i));
    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(input0_ptr);
  tensor_free<policy_t>(output0_ptr);

}



TYPED_TEST_P(TestTensorMatrix, ET_Transpose)
{
  ET_TransposeImpl<TypeParam>();
}


#endif
