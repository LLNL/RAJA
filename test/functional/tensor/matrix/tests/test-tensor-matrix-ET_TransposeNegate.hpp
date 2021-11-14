//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_TESNOR_MATRIX_ET_TransposeNegate_HPP__
#define __TEST_TESNOR_MATRIX_ET_TransposeNegate_HPP__

#include<RAJA/RAJA.hpp>

template <typename MATRIX_TYPE>
void ET_TransposeNegateImpl()
{

  using matrix_t = MATRIX_TYPE;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;
  static const int N = matrix_t::register_type::s_num_elem * 4;

  // Create a row-major data buffer
  element_t data1[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i + j*N;
    }
  }

  // Create an empty result bufffer
  element_t data2[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));

  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));



  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform transpose of view1 into view2
  view2(Row::all(), Col::all()) = -view1(Row::all(), Col::all()).transpose();


  // Check
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      //ASSERT_SCALAR_EQ(data2[i][j], -data1[j][i]);
    }
  }

}



TYPED_TEST_P(TestTensorMatrix, ET_TransposeNegate)
{
  ET_TransposeNegateImpl<TypeParam>();
}


#endif
