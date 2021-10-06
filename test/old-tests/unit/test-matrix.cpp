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
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4, RAJA::cuda_warp_register>
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

/*
 * We are using ((double)rand()/RAND_MAX) for input values so the compiler cannot do fancy
 * things, like constexpr out all of the intrinsics.
 */

GPU_TYPED_TEST_P(MatrixTest, Ctor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Data
  //
  std::vector<element_t> data1_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);


  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);



  //
  // Do Operation: broadcast-ctor and copy-ctor
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // create a matrix that contains all 3's
    matrix_t m1(element_t(3));

    // copy to another matrix
    matrix_t m2(m1);

    // write out both matrices
    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        data1_d(i,j) = m1.get(i,j);
        data2_d(i,j) = m2.get(i,j);
      }
    }

  });

  // copy data back to host
  tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);
  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(3, data1_h(i,j));
      ASSERT_SCALAR_EQ(3, data2_h(i,j));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);

}



GPU_TYPED_TEST_P(MatrixTest, Load_RowMajor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);



  // Fill data
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Full load
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    matrix_t m;
    if(matrix_t::layout_type::is_row_major()){
      m.load_packed(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }
    else{
      m.load_strided(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }

    // write out to a second view so we can check it on the host
    // on GPU's we'll write way too much, but it should stil be correct
    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        data2_d(i,j) = m.get(i,j);
      }
    }

  });

  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
          data2_h(i,j) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Partial load
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        matrix_t m;
        if(matrix_t::layout_type::is_row_major()){
          m.load_packed_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }
        else{
          m.load_strided_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }

        // write out to a second view so we can check it on the host
        // on GPU's we'll write way too much, but it should stil be correct
        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            data2_d(i,j) = m.get(i,j);
          }
        }

      });

      tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(0), data2_h(i,j));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
}




GPU_TYPED_TEST_P(MatrixTest, Load_ColMajor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  //
  // Allocate Row-Major data
  //

  // alloc data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_columns, 2*matrix_t::s_num_rows);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_columns, 2*matrix_t::s_num_rows);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // Fill data
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(j,i) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do operation
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
    matrix_t m;

    if(matrix_t::layout_type::is_column_major()){
      m.load_packed(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }
    else{
      m.load_strided(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }

    // write out to a second view so we can check it on the host
    // on GPU's we'll write way too much, but it should stil be correct
    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        data2_d(j,i) = m.get(i,j);
      }
    }

  });

  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(j,i), data2(j,i));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
          data2_h(j,i) = -1;
        }
      }

      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Partial load
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        matrix_t m;
        if(matrix_t::layout_type::is_column_major()){
          m.load_packed_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }
        else{
          m.load_strided_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }

        // write out to a second view so we can check it on the host
        // on GPU's we'll write way too much, but it should stil be correct
        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            data2_d(j,i) = m.get(i,j);
          }
        }

      });

      tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(0), data2_h(j,i));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
}



GPU_TYPED_TEST_P(MatrixTest, Store_RowMajor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  //
  // Allocate Row-Major Data
  //

  // alloc data1 - matrix data will be generated on device, stored into data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2 - reference data to compare with data1 on host

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);


  //
  // Fill reference data
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data2_h(i,j) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  //
  // Clear data1
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = element_t(-2);
    }
  }
  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Full store
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill out matrix
    matrix_t m(-1.0);

    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        m.set(2*i*matrix_t::s_num_columns+j, i, j);
      }
    }

    // Store matrix to memory
    if(matrix_t::layout_type::is_row_major()){
      m.store_packed(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }
    else{
      m.store_strided(data1_ptr, 2*matrix_t::s_num_columns, 1);
    }
  });

  tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      if(i < matrix_t::s_num_rows && j < matrix_t::s_num_columns){
//        printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
      }
      else{
//        printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(i,j), element_t(-2));
      }
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){

      //
      // Clear data1
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          data1_h(i,j) = element_t(-2);
        }
      }
      tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


      //
      // Do Operation: Partial Store
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // fill out matrix
        matrix_t m(-1.0);

        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            m.set(2*i*matrix_t::s_num_columns+j, i, j);
          }
        }

        // Store matrix to memory
        if(matrix_t::layout_type::is_row_major()){
          m.store_packed_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }
        else{
          m.store_strided_nm(data1_ptr, 2*matrix_t::s_num_columns, 1, n_size, m_size);
        }

      });


      tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          if(i < n_size && j < m_size){
//            printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(i,j));
          }
          else{
//            printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(i,j), element_t(-2));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
}


GPU_TYPED_TEST_P(MatrixTest, Store_ColMajor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  //
  // Allocate Column-Major Data
  //

  // alloc data1 - matrix data will be generated on device, stored into data1

  std::vector<element_t> data1_vec(4*matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), 2*matrix_t::s_num_columns, 2*matrix_t::s_num_rows);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2 - reference data to compare with data1 on host

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);


  //
  // Fill reference data
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data2_h(j,i) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  //
  // Clear data1
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(j,i) = element_t(-2);
    }
  }
  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Full store
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    // fill out matrix
    matrix_t m(-1.0);

    for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
      for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
        m.set(2*i*matrix_t::s_num_columns+j, i, j);
      }
    }

    // Store matrix to memory
    if(matrix_t::layout_type::is_column_major()){
      m.store_packed(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }
    else{
      m.store_strided(data1_ptr, 1, 2*matrix_t::s_num_rows);
    }
  });

  tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      if(i < matrix_t::s_num_rows && j < matrix_t::s_num_columns){
//        printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
      }
      else{
//        printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
        ASSERT_SCALAR_EQ(data1_h(j,i), element_t(-2));
      }
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){

      //
      // Clear data1
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          data1_h(j,i) = element_t(-2);
        }
      }
      tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


      //
      // Do Operation: Partial Store
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // fill out matrix
        matrix_t m(-1.0);

        for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
          for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
            m.set(2*i*matrix_t::s_num_columns+j, i, j);
          }
        }

        // Store matrix to memory
        if(matrix_t::layout_type::is_column_major()){
          m.store_packed_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }
        else{
          m.store_strided_nm(data1_ptr, 1, 2*matrix_t::s_num_rows, n_size, m_size);
        }

      });


      tensor_copy_to_host<policy_t>(data1_vec, data1_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
          if(i < n_size && j < m_size){
//            printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1_h(i,j), data2_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(j,i), data2_h(j,i));
          }
          else{
//            printf("%d,%d:  %lf, -2\n", (int)i, (int)j, data1_h(i,j));
            ASSERT_SCALAR_EQ(data1_h(j,i), element_t(-2));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
}





GPU_TYPED_TEST_P(MatrixTest, ET_LoadStore)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);



  // Fill data
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);


  //
  // Do Operation: Load/Store full matrix from one view to another
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    data2_d(cols, rows) = data1_d(rows, cols);

  });

  tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= matrix_t::s_num_rows; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= matrix_t::s_num_columns; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
          data2_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Load/Store partial matrix from one view to another
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        data2_d(cols, rows) = data1_d(rows, cols);
      });

      tensor_copy_to_host<policy_t>(data2_vec, data2_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
        for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data1_h(i,j), data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data2_h(j,i));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
}


GPU_TYPED_TEST_P(MatrixTest, ET_Add)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2

  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3

  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
      data2_h(i,j) = 1+i+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: Full sum of data1 and data2
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    data3_d(cols, rows) = data1_d(rows, cols) + data2_d(cols, rows);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)+data2_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data2_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Perform partial sum
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        data3_d(cols, rows) = data1_d(rows, cols) + data2_d(cols, rows);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)+data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data2_h(j,i));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}


GPU_TYPED_TEST_P(MatrixTest, ET_Subtract)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2

  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3

  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
      data2_h(i,j) = 1+i+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: Full sum of data1 and data2
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    data3_d(cols, rows) = data1_d(rows, cols) - data2_d(cols, rows);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)-data2_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data2_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Perform partial sum
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        data3_d(cols, rows) = data1_d(rows, cols) - data2_d(cols, rows);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)-data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data2_h(j,i));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}


GPU_TYPED_TEST_P(MatrixTest, ET_Divide)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;

  //
  // Allocate Row-Major Data
  //

  // alloc data1

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2

  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3

  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = i*matrix_t::s_num_columns+j;
      data2_h(i,j) = 1+i+j;
    }
  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: Full sum of data1 and data2
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    data3_d(cols, rows) = data1_d(rows, cols) / data2_d(cols, rows);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)/data2_h(j,i));
//      printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
    }
  }



  //
  // Loop over all possible sub-matrix sizes using the load_*_nm routines
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data2
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data2_h(j,i) = -1;
        }
      }
      tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


      //
      // Do Operation: Perform partial sum
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        data3_d(cols, rows) = data1_d(rows, cols) / data2_d(cols, rows);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
//          printf("%d,%d:  %lf, %lf\n", (int)i, (int)j, data1(i,j), data2(i,j));
          if(i < n_size && j < m_size){
            ASSERT_SCALAR_EQ(data3_h(j,i), data1_h(i,j)/data2_h(j,i));
          }
          else{
            ASSERT_SCALAR_EQ(element_t(-1), data2_h(j,i));
          }
        }
      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}


GPU_TYPED_TEST_P(MatrixTest, ET_MatrixVector)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using cvector_t = typename matrix_t::column_vector_type;
  using rvector_t = typename matrix_t::row_vector_type;

//  static constexpr camp::idx_t N = 8; //matrix_t::s_num_rows*matrix_t::s_num_columns*2;
  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns)*2;
  //
  // Allocate Row-Major Data
  //

  // alloc data1 - The matrix

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2 - The input vector

  std::vector<element_t> data2_vec(N);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data2_h(data2_vec.data(),  N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data2_d(data2_ptr,  N);


  // alloc data3 - The output vector

  std::vector<element_t> data3_vec(N);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data3_h(data3_vec.data(),  N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<1,int,0>> data3_d(data3_ptr,  N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = 3+i*N+j;
    }
    data2_h(i) = i+1;
  }

//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data1_h(i,j));
//    }
//    printf("\n");
//  }


//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("data2[%d]=%lf\n", (int)i, (double)data2_h(i));
//  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: A*x
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    auto vrow = RAJA::VectorIndex<int, rvector_t>::all();
    auto vcol = RAJA::VectorIndex<int, cvector_t>::all();

    data3_d(vcol) = data1_d(rows, cols) * data2_d(vrow);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){

    element_t expected(0);
    for(camp::idx_t j = 0;j < N; ++ j){
      expected += data1_h(i,j)*data2_h(j);
    }
//    printf("i=%d, expected=%e, data3=%e\n", (int)i, (double)expected, (double)data3_h(i));

    ASSERT_SCALAR_EQ(expected, data3_h(i));
  }

//return;


  //
  // Loop over all possible sub-matrix sizes for A*x
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        data3_h(i) = 0;
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation (x')*A
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        auto vrow = RAJA::VectorIndex<int, rvector_t>::range(0, m_size);
        auto vcol = RAJA::VectorIndex<int, cvector_t>::range(0, n_size);

        data3_d(vcol) = data1_d(rows, cols) * data2_d(vrow);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t i = 0;i < n_size; ++ i){


        element_t expected(0);
        for(camp::idx_t j = 0;j < m_size; ++ j){
          expected += data1_h(i,j) * data2_h(j);
        }

        if(i >= n_size || m_size == 0){
          expected = 0;
        }

//        printf("i=%d, expected=%e, data3=%e\n", (int)i, (double)expected, (double)data3_h(i));
        ASSERT_SCALAR_EQ(expected, data3_h(i));

      }


    }
  }



  //
  // Do Operation: (x')*A
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto rows = RAJA::RowIndex<int, matrix_t>::all();
    auto cols = RAJA::ColIndex<int, matrix_t>::all();

    auto vrow = RAJA::VectorIndex<int, rvector_t>::all();
    auto vcol = RAJA::VectorIndex<int, cvector_t>::all();

    data3_d(vrow) =  data2_d(vcol) * data1_d(rows, cols);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


  //
  // Check results
  //
  for(camp::idx_t j = 0;j < N; ++ j){


    element_t expected(0);
    for(camp::idx_t i = 0;i < N; ++ i){
      expected += data2_h(i)*data1_h(i,j);
    }

    ASSERT_SCALAR_EQ(expected, data3_h(j));
//    printf("i=%d, data3=%lf, expected=%lf\n", (int)j, (double)data3_h(j), (double)expected);
  }




  //
  // Loop over all possible sub-matrix sizes for (x')*A
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t j = 0;j < N; ++ j){
        data3_h(j) = 0;
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation (x')*A
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){
        // Load data using a View
        auto rows = RAJA::RowIndex<int, matrix_t>::range(0, n_size);
        auto cols = RAJA::ColIndex<int, matrix_t>::range(0, m_size);

        auto vrow = RAJA::VectorIndex<int, rvector_t>::range(0, m_size);
        auto vcol = RAJA::VectorIndex<int, cvector_t>::range(0, n_size);

        data3_d(vrow) =  data2_d(vcol) * data1_d(rows, cols);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      for(camp::idx_t j = 0;j < N; ++ j){

        element_t expected(0);

        for(camp::idx_t i = 0;i < n_size; ++ i){
          expected += data2_h(i) * data1_h(i,j);
        }

        if(j >= m_size || n_size == 0){
          expected = 0;
        }

//        printf("j=%d, expected=%e, data3=%e\n", (int)j, (double)expected, (double)data3_h(j));
        ASSERT_SCALAR_EQ(expected, data3_h(j));

      }


    }
  }


  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}


GPU_TYPED_TEST_P(MatrixTest, ET_MatrixMatrixMultiply)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using A_matrix_t = matrix_t;
  using B_matrix_t = typename matrix_t::transpose_type;
  using C_matrix_t = typename matrix_t::product_type;

//  static constexpr camp::idx_t N = 8; //matrix_t::s_num_rows*matrix_t::s_num_columns*2;
  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns);
  //
  // Allocate Row-Major Data
  //

  // alloc data1 - The left matrix

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2 - The right matrix

  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3 - The result matrix

  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = 1+i*N+j;
      data2_h(i,j) = 3+i*N+j;
    }

  }

//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data1_h(i,j));
//    }
//    printf("\n");
//  }


//  printf("data2:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data2_h(i,j));
//    }
//    printf("\n");
//  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);


  //
  // Do Operation: A*B
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto A_rows = RAJA::RowIndex<int, A_matrix_t>::all();
    auto A_cols = RAJA::ColIndex<int, A_matrix_t>::all();

    auto B_rows = RAJA::RowIndex<int, B_matrix_t>::all();
    auto B_cols = RAJA::ColIndex<int, B_matrix_t>::all();

    auto C_rows = RAJA::RowIndex<int, C_matrix_t>::all();
    auto C_cols = RAJA::ColIndex<int, C_matrix_t>::all();

    data3_d(C_rows, C_cols) = data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);

//  printf("data3:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }

  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t expected(0);
      for(camp::idx_t k = 0;k < N; ++ k){
        expected += data1_h(i,k)*data2_h(k,j);
      }
//    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

      ASSERT_SCALAR_EQ(expected, data3_h(i,j));
//      data3_h(i,j) = expected;

    }
  }

//  printf("expected:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }


  //
  // Loop over all possible sub-matrix sizes for A*x
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data3_h(i, j) = 0;
        }
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation A*B
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

        auto A_rows = RAJA::RowIndex<int, A_matrix_t>::range(0, n_size);
        auto A_cols = RAJA::ColIndex<int, A_matrix_t>::range(0, m_size);

        auto B_rows = RAJA::RowIndex<int, B_matrix_t>::range(0, m_size);
        auto B_cols = RAJA::ColIndex<int, B_matrix_t>::range(0, n_size);

        auto C_rows = RAJA::RowIndex<int, C_matrix_t>::range(0, n_size);
        auto C_cols = RAJA::ColIndex<int, C_matrix_t>::range(0, n_size);

        data3_d(C_rows, C_cols) = data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      //
      // Check results
      //
      for(camp::idx_t i = 0;i < n_size; ++ i){
        for(camp::idx_t j = 0;j < n_size; ++ j){
          element_t expected(0);
          for(camp::idx_t k = 0;k < m_size; ++ k){
            expected += data1_h(i,k)*data2_h(k,j);
          }
    //    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

          ASSERT_SCALAR_EQ(expected, data3_h(i,j));
    //      data3_h(i,j) = expected;

        }
      }


    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}




GPU_TYPED_TEST_P(MatrixTest, ET_MatrixMatrixMultiplyAdd)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;

  using A_matrix_t = matrix_t;
  using B_matrix_t = typename matrix_t::transpose_type;
  using C_matrix_t = typename matrix_t::product_type;

//  static constexpr camp::idx_t N = 8; //matrix_t::s_num_rows*matrix_t::s_num_columns*2;
  static constexpr camp::idx_t N = RAJA::max<camp::idx_t>(matrix_t::s_num_rows, matrix_t::s_num_columns);
  //
  // Allocate Row-Major Data
  //

  // alloc data1 - The left matrix

  std::vector<element_t> data1_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), N, N);

  element_t *data1_ptr = tensor_malloc<policy_t>(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr,  N, N);


  // alloc data2 - The right matrix

  std::vector<element_t> data2_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(),  N, N);

  element_t *data2_ptr = tensor_malloc<policy_t>(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr,  N, N);


  // alloc data3 - The result matrix

  std::vector<element_t> data3_vec(N*N);
  RAJA::View<element_t, RAJA::Layout<2>> data3_h(data3_vec.data(),  N, N);

  element_t *data3_ptr = tensor_malloc<policy_t>(data3_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data3_d(data3_ptr,  N, N);



  // Fill data1 and data2
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      data1_h(i,j) = 1+i*N+j;
      data2_h(i,j) = 3+i*N+j;
      data3_h(i,j) = 5*i+13*j;
    }

  }

//  printf("data1:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data1_h(i,j));
//    }
//    printf("\n");
//  }


//  printf("data2:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data2_h(i,j));
//    }
//    printf("\n");
//  }

  tensor_copy_to_device<policy_t>(data1_ptr, data1_vec);
  tensor_copy_to_device<policy_t>(data2_ptr, data2_vec);
  tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


  //
  // Do Operation: A*B
  //
  tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

    auto A_rows = RAJA::RowIndex<int, A_matrix_t>::all();
    auto A_cols = RAJA::ColIndex<int, A_matrix_t>::all();

    auto B_rows = RAJA::RowIndex<int, B_matrix_t>::all();
    auto B_cols = RAJA::ColIndex<int, B_matrix_t>::all();

    auto C_rows = RAJA::RowIndex<int, C_matrix_t>::all();
    auto C_cols = RAJA::ColIndex<int, C_matrix_t>::all();

    data3_d(C_rows, C_cols) += data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);

  });

  tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);

//  printf("data3:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }

  //
  // Check results
  //
  for(camp::idx_t i = 0;i < N; ++ i){
    for(camp::idx_t j = 0;j < N; ++ j){
      element_t expected(5*i+13*j);
      for(camp::idx_t k = 0;k < N; ++ k){
        expected += data1_h(i,k)*data2_h(k,j);
      }
//    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

      ASSERT_SCALAR_EQ(expected, data3_h(i,j));
//      data3_h(i,j) = expected;

    }
  }

//  printf("expected:\n");
//  for(camp::idx_t i = 0;i < N; ++ i){
//    printf("  ");
//    for(camp::idx_t j = 0;j < N; ++ j){
//      printf("%lf  ", (double)data3_h(i,j));
//    }
//    printf("\n");
//  }


  //
  // Loop over all possible sub-matrix sizes for A*x
  //
  for(camp::idx_t n_size = 0;n_size <= N; ++ n_size){
    for(camp::idx_t m_size = 0;m_size <= N; ++ m_size){
//      printf("Running %d x %d\n", (int)n_size, (int)m_size);
      //
      // Clear data3
      //
      for(camp::idx_t i = 0;i < N; ++ i){
        for(camp::idx_t j = 0;j < N; ++ j){
          data3_h(i,j) = 5*i+13*j;
        }
      }

      tensor_copy_to_device<policy_t>(data3_ptr, data3_vec);


      //
      // Do Operation A*B
      //
      tensor_do<policy_t>([=] RAJA_HOST_DEVICE (){

        auto A_rows = RAJA::RowIndex<int, A_matrix_t>::range(0, n_size);
        auto A_cols = RAJA::ColIndex<int, A_matrix_t>::range(0, m_size);

        auto B_rows = RAJA::RowIndex<int, B_matrix_t>::range(0, m_size);
        auto B_cols = RAJA::ColIndex<int, B_matrix_t>::range(0, n_size);

        auto C_rows = RAJA::RowIndex<int, C_matrix_t>::range(0, n_size);
        auto C_cols = RAJA::ColIndex<int, C_matrix_t>::range(0, n_size);

        data3_d(C_rows, C_cols) += data1_d(A_rows, A_cols) * data2_d(B_rows, B_cols);
      });

      tensor_copy_to_host<policy_t>(data3_vec, data3_ptr);


      //
      // Check results
      //
      //
      // Check results
      //
      for(camp::idx_t i = 0;i < n_size; ++ i){
        for(camp::idx_t j = 0;j < n_size; ++ j){
          element_t expected(5*i+13*j);
          for(camp::idx_t k = 0;k < m_size; ++ k){
            expected += data1_h(i,k)*data2_h(k,j);
          }
    //    printf("i=%d, j=%d, expected=%e, data3=%e\n", (int)i, (int)j, (double)expected, (double)data3_h(i,j));

          ASSERT_SCALAR_EQ(expected, data3_h(i,j));
    //      data3_h(i,j) = expected;

        }
      }


    }
  }



  //
  // Free data
  //
  tensor_free<policy_t>(data1_ptr);
  tensor_free<policy_t>(data2_ptr);
  tensor_free<policy_t>(data3_ptr);

}

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


REGISTER_TYPED_TEST_SUITE_P(MatrixTest,
                                        Ctor,
                                        Load_RowMajor,
                                        Load_ColMajor,
                                        Store_RowMajor,
                                        Store_ColMajor,

                                        ET_LoadStore,
                                        ET_Add,
                                        ET_Subtract,
                                        ET_Divide,
                                        ET_MatrixVector,
                                        ET_MatrixMatrixMultiply,
                                        ET_MatrixMatrixMultiplyAdd
//                                        ET_TransposeNegate
                                        );

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, MatrixTest, MatrixTestTypes);






