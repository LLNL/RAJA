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


//
//TEST(TensorMatrix, MatrixRegisterMap)
//{
//  RAJA::internal::MatrixRegisterMap<RAJA::ColMajorLayout, 4,4, 16> m1;
//
//  printf("4x4 in 16-elem register:\n");
//  printf("register map:\n");
//  for(int row = 0;row < 4;++ row){
//    for(int col = 0;col < 4;++ col){
//      printf("%2d ", (int)m1.to_register(row, col));
//    }
//    printf("\n");
//  }
//
//  printf("lane map:\n");
//  for(int row = 0;row < 4;++ row){
//    for(int col = 0;col < 4;++ col){
//      printf("%2d ", (int)m1.to_lane(row, col));
//    }
//    printf("\n");
//  }
//  printf("\n");
//
//  RAJA::internal::MatrixRegisterMap<RAJA::ColMajorLayout, 4,8, 8> m2;
//
//  printf("4x8 in 8-elem register:\n");
//  printf("register map:\n");
//  for(int row = 0;row < 4;++ row){
//    for(int col = 0;col < 8;++ col){
//      printf("%2d ", (int)m2.to_register(row, col));
//    }
//    printf("\n");
//  }
//
//  printf("lane map:\n");
//  for(int row = 0;row < 4;++ row){
//    for(int col = 0;col < 8;++ col){
//      printf("%2d ", (int)m2.to_lane(row, col));
//    }
//    printf("\n");
//  }
//  printf("\n");
//
//
//  RAJA::internal::MatrixRegisterMap<RAJA::ColMajorLayout, 8,4, 8> m3;
//
//  printf("8x4 in 8-elem register:\n");
//  printf("register map:\n");
//  for(int row = 0;row < 8;++ row){
//    for(int col = 0;col < 4;++ col){
//      printf("%2d ", (int)m3.to_register(row, col));
//    }
//    printf("\n");
//  }
//
//  printf("lane map:\n");
//  for(int row = 0;row < 8;++ row){
//    for(int col = 0;col < 4;++ col){
//      printf("%2d ", (int)m3.to_lane(row, col));
//    }
//    printf("\n");
//  }
//  printf("\n");
//}


using MatrixTestTypes = ::testing::Types<

//    // These tests use the platform default SIMD architecture
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4, 2>
RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2, 4>
//    RAJA::SquareMatrixRegister<float, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<float, RAJA::RowMajorLayout>,
//    RAJA::SquareMatrixRegister<long, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<long, RAJA::RowMajorLayout>,
//    RAJA::SquareMatrixRegister<int, RAJA::ColMajorLayout>,
//    RAJA::SquareMatrixRegister<int, RAJA::RowMajorLayout>,
//
//    // Tests tests force the use of scalar math
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout, RAJA::scalar_register>,
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout, RAJA::scalar_register>

  >;


template <typename NestedPolicy>
class MatrixTest : public ::testing::Test
{
protected:

  MatrixTest() = default;
  virtual ~MatrixTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};
TYPED_TEST_SUITE_P(MatrixTest);

#if 1
/*
 * We are using ((double)rand()/RAND_MAX) for input values so the compiler cannot do fancy
 * things, like constexpr out all of the intrinsics.
 */

TYPED_TEST_P(MatrixTest, MatrixCtor)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  matrix_t m_empty;
  ASSERT_SCALAR_EQ(m_empty.get(0,0), element_t(0));

  matrix_t m_bcast(element_t(1));
  ASSERT_SCALAR_EQ(m_bcast.get(0,0), element_t(1));

  matrix_t m_copy(m_bcast);
  ASSERT_SCALAR_EQ(m_copy.get(0,0), element_t(1));


}

TYPED_TEST_P(MatrixTest, MatrixGetSet)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  matrix_t m;
  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
      m.set(element_t(NO_OPT_ZERO + i+j*j), i,j);
      ASSERT_SCALAR_EQ(m.get(i,j), element_t(i+j*j));
    }
  }

  // Use assignment operator
  matrix_t m2;
  m2 = m;

  // Check values are same as m
  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m2.get(i,j), element_t(i+j*j));
    }
  }

}

TYPED_TEST_P(MatrixTest, MatrixLoad)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  // Row-Major data
  element_t data1[matrix_t::s_num_rows][matrix_t::s_num_columns];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data1[i][j] = i*j*j;
    }
  }

  // Load data
  matrix_t m1;
  if(matrix_t::layout_type::is_row_major()){
    m1.load_packed(&data1[0][0], matrix_t::s_num_rows, 1);
  }
  else{
    m1.load_strided(&data1[0][0], matrix_t::s_num_rows, 1);
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m1.get(i,j), data1[i][j]);
    }
  }



  // Column-Major data
  element_t data2[matrix_t::s_num_columns][matrix_t::s_num_rows];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data2[j][i] = i*j*j;
    }
  }

  // Load data
  matrix_t m2;
  if(matrix_t::layout_type::is_column_major()){
    m2.load_packed(&data2[0][0], 1, matrix_t::s_num_rows);
  }
  else{
    m2.load_strided(&data2[0][0], 1, matrix_t::s_num_rows);
  }
  printf("m2=%s\n", m2.to_string().c_str());

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m2.get(i,j), data2[j][i]);
    }
  }


}



TYPED_TEST_P(MatrixTest, MatrixStore)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;


  // Fill data
  matrix_t m;
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      m.set(i*matrix_t::s_num_columns + j, i,j);
    }
  }



  // Store to a Row-Major data buffer
  element_t data1[matrix_t::s_num_rows][matrix_t::s_num_columns];
  if(matrix_t::layout_type::is_row_major()){
    printf("store_packed\n");
    m.store_packed(&data1[0][0], matrix_t::s_num_rows, 1);
  }
  else{
    printf("store_strided\n");
    m.store_strided(&data1[0][0], matrix_t::s_num_rows, 1);
  }

  // Check contents
  printf("m=%s\n", m.to_string().c_str());
  printf("data1:\n");
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      printf("%lf ", data1[i][j]);
    }
    printf("\n");
  }
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data1[i][j]);
    }
  }

  // Store to a Column-Major data buffer
  element_t data2[matrix_t::s_num_columns][matrix_t::s_num_rows];

  if(matrix_t::layout_type::is_column_major()){
    printf("store_packed\n");

    m.store_packed(&data2[0][0], 1, matrix_t::s_num_rows);
  }
  else{
    printf("store_strided\n");

    m.store_strided(&data2[0][0], 1, matrix_t::s_num_rows);
  }

  printf("data2:\n");
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      printf("%lf ", data2[j][i]);
    }
    printf("\n");
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data2[j][i]);
    }
  }


}


TYPED_TEST_P(MatrixTest, MatrixViewLoad)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  // Row-Major data
  element_t data1[matrix_t::s_num_rows][matrix_t::s_num_columns];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data1[i][j] = i*matrix_t::s_num_columns + j;
    }
  }

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(&data1[0][0], matrix_t::s_num_rows, matrix_t::s_num_columns);

  // Load data
  auto rows = RAJA::RowIndex<int, matrix_t>::all();
  auto cols = RAJA::ColIndex<int, matrix_t>::all();
  matrix_t m1 = view1(rows, cols);

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m1.get(i,j), data1[i][j]);
    }
  }


  // Column-Major data
  element_t data2[matrix_t::s_num_columns][matrix_t::s_num_rows];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data2[j][i] = i*matrix_t::s_num_columns + j;
    }
  }

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{matrix_t::s_num_rows, matrix_t::s_num_columns}}, {{1,0}}));

  // Load data
  matrix_t m2 = view2(rows, cols);

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m2.get(i,j), data2[j][i]);
    }
  }

}

TYPED_TEST_P(MatrixTest, MatrixViewStore)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;


  // Fill data
  matrix_t m;
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      m.set(i*j*j, i,j);
    }
  }



  // Create a Row-Major data buffer
  element_t data1[matrix_t::s_num_rows][matrix_t::s_num_columns];

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(&data1[0][0], matrix_t::s_num_rows, matrix_t::s_num_columns);

  // Store using view
  RAJA::RowIndex<int, matrix_t> rows(0, matrix_t::s_num_rows);
  RAJA::ColIndex<int, matrix_t> cols(0, matrix_t::s_num_columns);
  view1(rows, cols) = m;

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data1[i][j]);
    }
  }




  // Create a Column-Major data buffer
  element_t data2[matrix_t::s_num_columns][matrix_t::s_num_rows];

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{matrix_t::s_num_rows, matrix_t::s_num_columns}}, {{1,0}}));

  // Store using view
  view2(rows, cols) = m;

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data2[j][i]);
    }
  }


}


TYPED_TEST_P(MatrixTest, MatrixVector)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;
  using col_vector_t = typename matrix_t::column_vector_type;
  using row_vector_t = typename matrix_t::row_vector_type;
  static constexpr camp::idx_t num_rows = matrix_t::s_num_rows;
  static constexpr camp::idx_t num_columns = matrix_t::s_num_columns;

  // initialize a matrix and vector
  matrix_t m;
  for(camp::idx_t j = 0;j < num_columns; ++ j){
    for(camp::idx_t i = 0;i < num_rows; ++ i){
      m.set(element_t(NO_OPT_ZERO + 5+i+j*j), i,j);
    }
  }


  {
    col_vector_t v;
    for(camp::idx_t i = 0;i < num_rows; ++ i){
      v.set(NO_OPT_ZERO + 3 + i*2, i);
    }


    // matrix vector product
    // note mv is not necessarily the same type as v
    auto mv = m.right_multiply_vector(v);

    printf("m: %s", m.to_string().c_str());
    printf("v: %s", v.to_string().c_str());
    printf("mv: %s", mv.to_string().c_str());

    // check result
    for(camp::idx_t i = 0;i < num_rows; ++ i){
      element_t expected(0);

      for(camp::idx_t j = 0;j < num_columns; ++ j){
        expected += m.get(i,j)*v.get(j);
      }

      printf("mv: i=%d, val=%lf, %s", (int)i, (double)mv.get(0), mv.to_string().c_str());

      ASSERT_SCALAR_EQ(mv.get(i), expected);
    }
  }

//  {
//
//    row_vector_t v;
//    for(camp::idx_t j = 0;j < num_columns; ++ j){
//      v.set(NO_OPT_ZERO + 3 + j*2, j);
//    }
//
//    // matrix vector product
//    auto mv = m.left_multiply_vector(v);
//
//    // check result
//    for(camp::idx_t j = 0;j < num_columns; ++ j){
//      element_t expected(0);
//
//      for(camp::idx_t i = 0;i < num_rows; ++ i){
//        expected += m.get(i,j)*v.get(i);
//      }
//
//      ASSERT_SCALAR_EQ(mv.get(j), expected);
//    }
//  }
}

#if 0

TYPED_TEST_P(MatrixTest, MatrixMatrix)
{

  using A_t = TypeParam;
  using B_t = TypeParam;
  using element_t = typename A_t::element_type;

  camp::idx_t size_a0 = 1;
  camp::idx_t size_b0 = 1;


  // Loop over different sizes of matrices A and B
  //
  // A = size_a x size_b
  // B = size_b x size_a
  // C = size_a x size_a
  for(camp::idx_t size_a = size_a0;size_a <= A_t::register_type::s_num_elem;size_a ++){
    for(camp::idx_t size_b = size_b0;size_b <= A_t::register_type::s_num_elem;size_b ++){

      // initialize two matrices
      A_t A;
      A.clear();

      for(camp::idx_t j = 0;j < size_b; ++ j){
        for(camp::idx_t i = 0;i < size_a; ++ i){
          A.set(element_t(NO_OPT_ZERO + i+j*j), i,j);
        }
      }

      B_t B;
      B.clear();
      for(camp::idx_t j = 0;j < size_a; ++ j){
        for(camp::idx_t i = 0;i < size_b; ++ i){
          B.set(element_t(NO_OPT_ZERO + i*i+2*j), i,j);
        }
      }


      // matrix matrix product
      auto C = A*B;


      // check result
      for(camp::idx_t i = 0;i < size_a; ++ i){

        for(camp::idx_t j = 0;j < size_a; ++ j){

          // do dot product to compute C(i,j)
          element_t expected(0);
          for(camp::idx_t k = 0;k < size_b; ++ k){
            expected += A.get(i, k) * B.get(k,j);
          }

          ASSERT_SCALAR_EQ(C.get(i,j), expected);
        }
      }


    }
  }



}
TYPED_TEST_P(MatrixTest, MatrixMatrixAccumulate)
{

  using matrix_t = TypeParam;

  using element_t = typename matrix_t::element_type;

  // initialize two matrices
  matrix_t A;
  for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
      A.set(element_t(NO_OPT_ZERO + i+j*j), i,j);
    }
  }

  matrix_t B;
  for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
      if(i == 0){
        B.set(element_t(0), i, j);
      }
      else{
        B.set(element_t(NO_OPT_ZERO + i*i+j*j), i, j);
      }

    }
  }

  using C_t = decltype(A*B);

  C_t C;
  for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
      C.set(element_t(NO_OPT_ZERO + 2*i+3*j), i, j);
    }
  }

//  printf("A:\n%s\n", A.toString().c_str());
//  printf("B:\n%s\n", B.toString().c_str());
//  printf("C:\n%s\n", C.toString().c_str());

  // matrix matrix product
  auto Z1 = A*B+C;

//  printf("Z1:\n%s\n", Z1.toString().c_str());


  // explicit
  auto Z2 = A.matrix_multiply_add(B, C);

//  printf("Z2:\n%s\n", Z2.toString().c_str());

//  // check result
//  decltype(Z1) expected;
//  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
//
//    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
//
//      // do dot product to compute C(i,j)
//      element_t z = C(i,j);
//      for(camp::idx_t k = 0;k < matrix_t::register_type::s_num_elem; ++ k){
//        z += A.get(i, k) * B(k,j);
//      }
//
//      expected.set(z, i,j);
//    }
//  }
//  printf("Expected:\n%s\n", expected.toString().c_str());


  // check result
  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){

    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){

      // do dot product to compute C(i,j)
      element_t expected = C.get(i,j);
      for(camp::idx_t k = 0;k < matrix_t::register_type::s_num_elem; ++ k){
        expected += A.get(i, k) * B.get(k,j);
      }

      ASSERT_SCALAR_EQ(Z1.get(i,j), expected);
      ASSERT_SCALAR_EQ(Z2.get(i,j), expected);
    }
  }

}


TYPED_TEST_P(MatrixTest, MatrixTranspose)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static constexpr camp::idx_t num_elem = matrix_t::register_type::s_num_elem;

  matrix_t m;
//  printf("M:\n");
  for(camp::idx_t i = 0;i < num_elem; ++ i){
    for(camp::idx_t j = 0;j < num_elem; ++ j){
      m.set(element_t(i+j*num_elem), i,j);
//      printf("%3lf ", (double)m.get(i,j));
    }
//    printf("\n");
  }

  // Use transpose.. keeping matrix layout and transposing data
  matrix_t mt = m.transpose();

  // Check values are transposed
  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(mt.get(j,i), element_t(i+j*num_elem));
    }
  }


  // Use transpose_type.. swaps data layout, keeping data in place
  auto mt2 = m.transpose_type();

  // Check values are transposed
  for(camp::idx_t i = 0;i < matrix_t::register_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::register_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(mt2.get(j,i), element_t(i+j*num_elem));
    }
  }
}

TYPED_TEST_P(MatrixTest, ETLoadStore)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::register_type::s_num_elem * 16;

  // Create a row-major data buffer
  element_t data1[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
    }
  }

  // Create an empty data bufffer
  element_t data2[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{1,0}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{1,0}}));


  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform copy of view1 into view2
  view2(Row::all(), Col::all()) = view1(Row::all(), Col::all());


  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[i][j]);
    }
  }

  // Perform transpose view1 into view2 by switching col and row for view1
  view2(Row::all(), Col::all()) = view1(Col::all(), Row::all());

  // Check that data1==transpose(data2)
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[j][i]);
    }
  }

  // Perform transpose view1 into view2 by switching col and row for view2
  view2(Col::all(), Row::all()) = view1(Row::all(), Col::all());

  // Check that data1==transpose(data2)
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data1[i][j], data2[j][i]);
    }
  }

}

TYPED_TEST_P(MatrixTest, ETAdd)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::register_type::s_num_elem * 4;

  // Create a row-major data buffer
  element_t data1[N][N], data2[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
      data2[i][j] = i+2*j;
    }
  }

  // Create an empty result bufffer
  element_t data3[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));

  RAJA::View<element_t, RAJA::Layout<2, int>> view3(
      &data3[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform copy of view1 into view2
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) +
                                  view2(Row::all(), Col::all());



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data1[i][j] + data2[i][j];

      ASSERT_SCALAR_EQ(data3[i][j], result);

      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

}

TYPED_TEST_P(MatrixTest, ETSubtract)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::register_type::s_num_elem * 4;

  // Create a row-major data buffer
  element_t data1[N][N], data2[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
      data2[i][j] = i+2*j;
    }
  }

  // Create an empty result bufffer
  element_t data3[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));

  RAJA::View<element_t, RAJA::Layout<2, int>> view3(
      &data3[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));



  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform subtraction of view2 from  view1
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) -
                                  view2(Row::all(), Col::all());



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data1[i][j] - data2[i][j];

      ASSERT_SCALAR_EQ(data3[i][j], result);

      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

  using vector_t = typename matrix_t::column_vector_type;
  using Vec = RAJA::VectorIndex<int, vector_t>;

  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      view3(i,j) = 0;
    }
  }

  // Perform subtraction of view1 from  view2
  // but do it row-by-row
  for(camp::idx_t i = 0;i < N; ++ i){
    view3(i, Vec::all()) = view2(i, Vec::all()) - view1(i, Vec::all());
  }



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data2[i][j] - data1[i][j];

      ASSERT_SCALAR_EQ(data3[i][j], result);

//      printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }


  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      view3(i,j) = 0;
    }
  }

  // Perform subtraction of view1 from  view2
  // but do it column-by-column
  for(camp::idx_t i = 0;i < N; ++ i){
    view3(Vec::all(),i) = view2(Vec::all(),i) - view1(Vec::all(), i);
  }



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data2[i][j] - data1[i][j];

      ASSERT_SCALAR_EQ(data3[i][j], result);

//      printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

}

TYPED_TEST_P(MatrixTest, ETMatrixVectorMultiply)
{
  using matrix_t = TypeParam;
  using vector_t = typename matrix_t::column_vector_type;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::register_type::s_num_elem * 4;

  // Create a row-major data buffer
  element_t data1[N][N], data2[N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;

    }
    data2[i] = i*i;
  }

  // print
//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%e ", (double)data1[i][j]);
//    }
//    printf("\n");
//  }
//
//  printf("\n");
//  printf("data2:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("%e ", (double)data2[i]);
//  }
//  printf("\n");


  // Create an empty result bufffer
  element_t data3[N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  RAJA::View<element_t, RAJA::Layout<1, int>> view2(
      &data2[0], RAJA::make_permuted_layout<1, int>({{N}}, {{0}}));

  RAJA::View<element_t, RAJA::Layout<1, int>> view3(
      &data3[0], RAJA::make_permuted_layout<1, int>({{N}}, {{0}}));



  using Vec = RAJA::VectorIndex<int, vector_t>;
  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Performn right matrix-vector multiplication
  view3(Vec::all()) = view1(Row::all(), Col::all()) * view2(Vec::all());

  // Check
  for(camp::idx_t i = 0;i < N; ++ i){
    element_t result = 0;
    for(camp::idx_t j = 0;j < N; ++ j){
      result += data1[i][j] * data2[j];
    }

    ASSERT_SCALAR_EQ(data3[i], result);
  }




  // Perform left matrix-vector multiplication
  view3(Vec::all()) = view2(Vec::all()) * view1(Row::all(), Col::all());

  // Check
  for(camp::idx_t j = 0;j < N; ++ j){

    element_t result = 0;
    for(camp::idx_t i = 0;i < N; ++ i){
      result += data1[i][j] * data2[i];
    }

    ASSERT_SCALAR_EQ(data3[j], result);
//    printf("(%d): val=%e, exp=%e\n",(int)j, (double)data3[j], (double)result);
  }

}


TYPED_TEST_P(MatrixTest, ETMatrixMatrixMultiply)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::register_type::s_num_elem * 4;

  // Create a row-major data buffer
  element_t data1[N][N], data2[N][N];
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
      data2[i][j] = i+2*j;
    }
  }

  // Create an empty result bufffer
  element_t data3[N][N];

  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));

  RAJA::View<element_t, RAJA::Layout<2, int>> view3(
      &data3[0][0], RAJA::make_permuted_layout<2, int>({{N, N}}, {{0,1}}));


  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  view3(Row::all(), Col::all()) = 2.0* view1(Row::all(), Col::all()) *
                                  view2(Row::all(), Col::all()) / 2.0;



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = 0;

      for(camp::idx_t k = 0;k < N; ++ k){
        result += data1[i][k] * data2[k][j];
      }

      ASSERT_SCALAR_EQ(data3[i][j], result);
    }
  }

}


TYPED_TEST_P(MatrixTest, ETMatrixMatrixMultiplyAdd)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int Nmax = matrix_t::register_type::s_num_elem * 2;

  static const int N = Nmax;

  // Create a row-major data buffer
  element_t data1[Nmax][Nmax], data2[Nmax][Nmax];

  // Create an empty result bufffer
  element_t data3[Nmax][Nmax];



  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1[i][j] = i*j*j;
      data2[i][j] = i+2*j;
    }
  }



  //  Create views
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(
      &data1[0][0], RAJA::make_permuted_layout<2, int>({{Nmax, Nmax}}, {{0,1}}));


  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{Nmax, Nmax}}, {{0,1}}));

  RAJA::View<element_t, RAJA::Layout<2, int>> view3(
      &data3[0][0], RAJA::make_permuted_layout<2, int>({{Nmax, Nmax}}, {{0,1}}));


  using Row = RAJA::RowIndex<int, matrix_t>;
  using Col = RAJA::ColIndex<int, matrix_t>;


  // Perform view3 = 2.0 * view1 * view2 + view1;
  auto rows = Row::range(0,N);
  auto cols = Col::range(0,N);
  view3(rows, cols) = view1(rows, cols) * view2(rows, cols) + view1(rows, cols);


  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data1[i][j];

      for(camp::idx_t k = 0;k < N; ++ k){
        result += data1[i][k] * data2[k][j];
      }
      ASSERT_SCALAR_EQ(data3[i][j], result);
      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

  // Perform view3 = view1 + view2 * view1 * 2.0;
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) +
                                  view2(Row::all(), Col::all()) *
                                  view1(Row::all(), Col::all()) * 2.0;



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data1[i][j];

      for(camp::idx_t k = 0;k < N; ++ k){
        result += data2[i][k] * data1[k][j] * 2.0;
      }
      ASSERT_SCALAR_EQ(data3[i][j], result);
      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

  // Perform view3 = view1,
  //  and    view1 += view2 * view1;
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all());
  view3(Row::all(), Col::all()) += view2(Row::all(), Col::all()) *
                                   view1(Row::all(), Col::all());



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = data1[i][j];

      for(camp::idx_t k = 0;k < N; ++ k){
        result += data2[i][k] * data1[k][j];
      }
      ASSERT_SCALAR_EQ(data3[i][j], result);
      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

}



TYPED_TEST_P(MatrixTest, ETMatrixTransposeNegate)
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


REGISTER_TYPED_TEST_SUITE_P(MatrixTest,
                                          MatrixCtor,
                                          MatrixGetSet,
                                          MatrixLoad,
                                          MatrixStore,
                                          MatrixViewLoad,
                                          MatrixViewStore,
                                          MatrixVector
//                                          MatrixMatrix,
//                                          MatrixMatrixAccumulate,
//                                          MatrixTranspose,
//
//                                        ETLoadStore,
//                                        ETAdd,
//                                        ETSubtract,
//                                        ETMatrixVectorMultiply,
//                                        ETMatrixMatrixMultiply,
//                                        ETMatrixMatrixMultiplyAdd,
//                                        ETMatrixTransposeNegate
                                        );

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, MatrixTest, MatrixTestTypes);





#endif
