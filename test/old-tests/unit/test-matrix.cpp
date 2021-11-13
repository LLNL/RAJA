//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for basic simd/simt vector operations
///

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include "./tensor-helper.hpp"

using MatrixTestTypes = ::testing::Types<

#ifdef RAJA_ENABLE_CUDA
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,4, RAJA::cuda_warp_register>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,8, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4, RAJA::cuda_warp_register>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,8, RAJA::cuda_warp_register>,
#endif

//    // These tests use the platform default SIMD architecture
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout>,
//
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,2>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,8>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,2>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,4>,
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,8>,
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2,2>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4>
//
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 16,4>,
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 4,16>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4,8>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4, 4>,
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 4, 2>,
////
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 2, 4>,
//    RAJA::SquareMatrixRegister<float, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<float, RAJA::RowMajorLayout>
//    RAJA::SquareMatrixRegister<long, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<long, RAJA::RowMajorLayout>,
//    RAJA::SquareMatrixRegister<int, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<int, RAJA::RowMajorLayout>,
//
//    // Tests tests force the use of scalar math
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout, RAJA::scalar_register>,
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout, RAJA::scalar_register>

  >;



#if 0




TYPED_TEST_P(MatrixTest, ET_TransposeNegate)
{
  using matrix_t = TypeParam;
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
#endif








