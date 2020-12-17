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

using MatrixTestTypes = ::testing::Types<
    RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::avx2_register>//,

/*
#ifdef __AVX__
    RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<double, RAJA::MATRIX_ROW_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_COL_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_ROW_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_COL_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_ROW_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_COL_MAJOR, RAJA::avx_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_ROW_MAJOR, RAJA::avx_register>,
#endif

#ifdef __AVX2__
    RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<double, RAJA::MATRIX_ROW_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_COL_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_ROW_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_COL_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_ROW_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_COL_MAJOR, RAJA::avx2_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_ROW_MAJOR, RAJA::avx2_register>,
#endif

#ifdef __AVX512__
    RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<double, RAJA::MATRIX_ROW_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_COL_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_ROW_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_COL_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_ROW_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_COL_MAJOR, RAJA::avx512_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_ROW_MAJOR, RAJA::avx512_register>,
#endif

    RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<double, RAJA::MATRIX_ROW_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_COL_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<float, RAJA::MATRIX_ROW_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_COL_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<long, RAJA::MATRIX_ROW_MAJOR, RAJA::scalar_register>,
    RAJA::RegisterMatrix<int, RAJA::MATRIX_COL_MAJOR, RAJA::scalar_register>, */
    //RAJA::RegisterMatrix<int, RAJA::MATRIX_ROW_MAJOR, RAJA::scalar_register>

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

#if 0
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
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      m.set(i,j, element_t(NO_OPT_ZERO + i+j*j));
      ASSERT_SCALAR_EQ(m.get(i,j), element_t(i+j*j));
    }
  }

  // Use assignment operator
  matrix_t m2;
  m2 = m;

  // Check values are same as m
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m2.get(i,j), element_t(i+j*j));
    }
  }

}

TYPED_TEST_P(MatrixTest, MatrixLoad)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  // Row-Major data
  element_t data1[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      data1[i][j] = i*j*j;
    }
  }

  // Load data
  matrix_t m1;
  if(matrix_t::layout_type::is_row_major()){
    m1.load_packed(&data1[0][0], matrix_t::vector_type::s_num_elem, 1);
  }
  else{
    m1.load_strided(&data1[0][0], matrix_t::vector_type::s_num_elem, 1);
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m1.get(i,j), data1[i][j]);
    }
  }



  // Column-Major data
  element_t data2[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      data2[j][i] = i*j*j;
    }
  }

  // Load data
  matrix_t m2;
  if(matrix_t::layout_type::is_column_major()){
    m2.load_packed(&data2[0][0], 1, matrix_t::vector_type::s_num_elem);
  }
  else{
    m2.load_strided(&data2[0][0], 1, matrix_t::vector_type::s_num_elem);
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
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
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      m.set(i,j, i*j*j);
    }
  }



  // Store to a Row-Major data buffer
  element_t data1[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];
  if(matrix_t::layout_type::is_row_major()){
    m.store_packed(&data1[0][0], matrix_t::vector_type::s_num_elem, 1);
  }
  else{
    m.store_strided(&data1[0][0], matrix_t::vector_type::s_num_elem, 1);
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data1[i][j]);
    }
  }

  // Store to a Column-Major data buffer
  element_t data2[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  if(matrix_t::layout_type::is_column_major()){
    m.store_packed(&data2[0][0], 1, matrix_t::vector_type::s_num_elem);
  }
  else{
    m.store_strided(&data2[0][0], 1, matrix_t::vector_type::s_num_elem);
  }

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data2[j][i]);
    }
  }


}


TYPED_TEST_P(MatrixTest, MatrixViewLoad)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  // Row-Major data
  element_t data1[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      data1[i][j] = i*j*j;
    }
  }

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(&data1[0][0], matrix_t::vector_type::s_num_elem, matrix_t::vector_type::s_num_elem);

  // Load data
  matrix_t m1 = view1(RAJA::RowIndex<int, matrix_t>(0), RAJA::ColIndex<int, matrix_t>(0));

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m1.get(i,j), data1[i][j]);
    }
  }


  // Column-Major data
  element_t data2[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      data2[j][i] = i*j*j;
    }
  }

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{matrix_t::vector_type::s_num_elem, matrix_t::vector_type::s_num_elem}}, {{1,0}}));

  // Load data
  matrix_t m2 = view2(RAJA::RowIndex<int, matrix_t>(0), RAJA::ColIndex<int, matrix_t>(0));

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
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
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      m.set(i,j, i*j*j);
    }
  }



  // Create a Row-Major data buffer
  element_t data1[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view1(&data1[0][0], matrix_t::vector_type::s_num_elem, matrix_t::vector_type::s_num_elem);

  // Store using view
  view1(RAJA::RowIndex<int, matrix_t>(0), RAJA::ColIndex<int, matrix_t>(0)) = m;

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data1[i][j]);
    }
  }




  // Create a Column-Major data buffer
  element_t data2[matrix_t::vector_type::s_num_elem][matrix_t::vector_type::s_num_elem];

  // Create a view of the data
  RAJA::View<element_t, RAJA::Layout<2, int>> view2(
      &data2[0][0], RAJA::make_permuted_layout<2, int>({{matrix_t::vector_type::s_num_elem, matrix_t::vector_type::s_num_elem}}, {{1,0}}));

  // Store using view
  view2(RAJA::RowIndex<int, matrix_t>(0), RAJA::ColIndex<int, matrix_t>(0)) = m;

  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data2[j][i]);
    }
  }


}


TYPED_TEST_P(MatrixTest, MatrixVector)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;
  using vector_t = typename matrix_t::vector_type;

  // initialize a matrix and vector
  matrix_t m;
  vector_t v;
  for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
      m.set(i,j, element_t(NO_OPT_ZERO + i+j*j));
    }
    v.set(j, j*2);
  }

  // matrix vector product
  // note mv is not necessarily the same type as v
  auto mv = m*v;

  // check result
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
    element_t expected(0);

    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
      expected += m.get(i,j)*v[j];
    }

    ASSERT_SCALAR_EQ(mv[i], expected);
  }

}

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
  for(camp::idx_t size_a = size_a0;size_a <= A_t::vector_type::s_num_elem;size_a ++){
    for(camp::idx_t size_b = size_b0;size_b <= A_t::vector_type::s_num_elem;size_b ++){

      // initialize two matrices
      A_t A;
      A.clear();

      for(camp::idx_t j = 0;j < size_b; ++ j){
        for(camp::idx_t i = 0;i < size_a; ++ i){
          A.set(i,j, element_t(NO_OPT_ZERO + i+j*j));
        }
      }

      B_t B;
      B.clear();
      for(camp::idx_t j = 0;j < size_a; ++ j){
        for(camp::idx_t i = 0;i < size_b; ++ i){
          B.set(i,j, element_t(NO_OPT_ZERO + i*i+2*j));
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
            expected += A.get(i, k) * B(k,j);
          }

          ASSERT_SCALAR_EQ(C.get(i,j), expected);
        }
      }


    }
  }



}
#endif
TYPED_TEST_P(MatrixTest, MatrixMatrixAccumulate)
{

  using matrix_t = TypeParam;

  using element_t = typename matrix_t::element_type;

  // initialize two matrices
  matrix_t A;
  for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
      A.set(i,j, element_t(NO_OPT_ZERO + i+j*j));
    }
  }

  matrix_t B;
  for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
      if(i == 0){
        B.set(i,j, element_t(0));
      }
      else{
        B.set(i,j, element_t(NO_OPT_ZERO + i*i+j*j));
      }

    }
  }

  using C_t = decltype(A*B);

  C_t C;
  for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
    for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
      C.set(i,j, element_t(NO_OPT_ZERO + 2*i+3*j));
    }
  }

//  printf("A:\n%s\n", A.toString().c_str());
//  printf("B:\n%s\n", B.toString().c_str());
//  printf("C:\n%s\n", C.toString().c_str());

  // matrix matrix product
  auto Z1 = A*B+C;

//  printf("Z1:\n%s\n", Z1.toString().c_str());


  // explicit
  auto Z2 = A.multiply_accumulate(B, C);

//  printf("Z2:\n%s\n", Z2.toString().c_str());

//  // check result
//  decltype(Z1) expected;
//  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){
//
//    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){
//
//      // do dot product to compute C(i,j)
//      element_t z = C(i,j);
//      for(camp::idx_t k = 0;k < matrix_t::vector_type::s_num_elem; ++ k){
//        z += A.get(i, k) * B(k,j);
//      }
//
//      expected.set(i,j,z);
//    }
//  }
//  printf("Expected:\n%s\n", expected.toString().c_str());


  // check result
  for(camp::idx_t i = 0;i < matrix_t::vector_type::s_num_elem; ++ i){

    for(camp::idx_t j = 0;j < matrix_t::vector_type::s_num_elem; ++ j){

      // do dot product to compute C(i,j)
      element_t expected = C(i,j);
      for(camp::idx_t k = 0;k < matrix_t::vector_type::s_num_elem; ++ k){
        expected += A.get(i, k) * B(k,j);
      }

      ASSERT_SCALAR_EQ(Z1.get(i,j), expected);
      ASSERT_SCALAR_EQ(Z2.get(i,j), expected);
    }
  }

}


TYPED_TEST_P(MatrixTest, AllLoadStore)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::vector_type::s_num_elem * 16;

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

TYPED_TEST_P(MatrixTest, AllAdd)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::vector_type::s_num_elem * 4;

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


TYPED_TEST_P(MatrixTest, AllMultiply)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::vector_type::s_num_elem * 4;

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
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) *
                                  view2(Row::all(), Col::all());



  // Check that data1==data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t result = 0;

      for(camp::idx_t k = 0;k < N; ++ k){
        result += data1[i][k] * data2[k][j];
      }
      ASSERT_SCALAR_EQ(data3[i][j], result);
      //printf("(%d,%d): val=%e, exp=%e\n",(int)i, (int)j, (double)data3[i][j], (double)result);
    }
  }

}

TYPED_TEST_P(MatrixTest, AllMultiplyAdd)
{
  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  static const int N = matrix_t::vector_type::s_num_elem * 4;

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


  // Perform view3 = view1 * view2 + view1;
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) *
                                  view2(Row::all(), Col::all()) +
                                  view1(Row::all(), Col::all());



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

  // Perform view3 = view1 + view2 * view1;
  view3(Row::all(), Col::all()) = view1(Row::all(), Col::all()) +
                                  view2(Row::all(), Col::all()) *
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


REGISTER_TYPED_TEST_SUITE_P(MatrixTest,
//    MatrixCtor,
//                                        MatrixGetSet,
//                                        MatrixLoad,
//                                        MatrixStore,
//                                        MatrixViewLoad,
//                                        MatrixViewStore,
//                                        MatrixVector,
//                                        MatrixMatrix,
                                        MatrixMatrixAccumulate,

                                          AllLoadStore,
                                          AllAdd,
                                          AllMultiply,
                                          AllMultiplyAdd);

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, MatrixTest, MatrixTestTypes);





