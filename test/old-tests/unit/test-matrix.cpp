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

#ifdef RAJA_ENABLE_CUDA
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,4, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4, RAJA::cuda_warp_register>,
#endif

//    // These tests use the platform default SIMD architecture
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout>
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout>,

//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,2>
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,8>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 2,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,2>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,8>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2,4>,
//
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 16,4>,
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<float, RAJA::ColMajorLayout, 4,16>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4,4>
//    RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4,8>
//      RAJA::RectMatrixRegister<float, RAJA::RowMajorLayout, 4, 4>
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4, 2>

//RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2, 4>
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


template<typename POL>
struct TensorTestHelper {

    template<typename BODY>
    static
    void exec(BODY const &body){
      body();
    }
};

#ifdef RAJA_ENABLE_CUDA

template <typename BODY>
__global__
void test_launcher(BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body();
}

template<>
struct TensorTestHelper<RAJA::cuda_warp_register>
{

    RAJA_SUPPRESS_HD_WARN
    template<typename BODY>
    static
    void exec(BODY const &body){
      cudaDeviceSynchronize();

      test_launcher<<<1,32>>>(body);

      cudaDeviceSynchronize();

    }
};
#endif


template<typename POL, typename BODY>
void tensor_do(BODY const &body){
  TensorTestHelper<POL>::exec(body);
}



#ifdef RAJA_ENABLE_CUDA

template<typename T>
T* tensor_malloc(size_t len){
  T *ptr;

  cudaErrchk(cudaMalloc(&ptr, len*sizeof(T)));

  return ptr;
}

template<typename T>
void tensor_free(T *ptr){
  cudaErrchk(cudaFree(ptr));
}

template<typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  cudaErrchk(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  cudaErrchk(cudaMemcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T), cudaMemcpyDeviceToHost));
}

#else

template<typename T>
T* tensor_malloc(size_t len){
  return new T[len];
}

template<typename T>
void tensor_free(T *ptr){
  delete[] ptr;
}

template<typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  memcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T));
}

template<typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  memcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T));
}

#endif



// Sugar to make things cleaner
template<typename T>
T* tensor_malloc(std::vector<T> const &vec){
  return tensor_malloc<T>(vec.size());
}



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

GPU_TYPED_TEST_P(MatrixTest, MatrixCtor)
{

  using matrix_t = TypeParam;
  using policy_t = typename matrix_t::register_policy;
  using element_t = typename matrix_t::element_type;


  //
  // Allocate Data
  //
  std::vector<element_t> data1_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data1_h(data1_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data1_ptr = tensor_malloc(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);


  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc(data2_vec);
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
  tensor_copy_to_host(data1_vec, data1_ptr);
  tensor_copy_to_host(data2_vec, data2_ptr);


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
  tensor_free(data1_ptr);
  tensor_free(data2_ptr);

}


#if 0

TYPED_TEST_P(MatrixTest, MatrixGetSet)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;

  matrix_t m;
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      m.set(element_t(NO_OPT_ZERO + i+j*j), i,j);
      ASSERT_SCALAR_EQ(m.get(i,j), element_t(i+j*j));
    }
  }

  // Use assignment operator
  matrix_t m2;
  m2 = m;

  // Check values are same as m
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m2.get(i,j), element_t(i+j*j));
    }
  }

}

#endif

GPU_TYPED_TEST_P(MatrixTest, MatrixLoad_RowMajor)
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

  element_t *data1_ptr = tensor_malloc(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_rows, 2*matrix_t::s_num_columns);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_rows, matrix_t::s_num_columns);

  element_t *data2_ptr = tensor_malloc(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_rows, matrix_t::s_num_columns);



  // Fill data
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(i,j) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device(data1_ptr, data1_vec);


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

  tensor_copy_to_host(data2_vec, data2_ptr);


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
      tensor_copy_to_device(data2_ptr, data2_vec);


      //
      // Do Operation: Full load
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

      tensor_copy_to_host(data2_vec, data2_ptr);


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
  tensor_free(data1_ptr);
  tensor_free(data2_ptr);
}




GPU_TYPED_TEST_P(MatrixTest, MatrixLoad_ColMajor)
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

  element_t *data1_ptr = tensor_malloc(data1_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data1_d(data1_ptr, 2*matrix_t::s_num_columns, 2*matrix_t::s_num_rows);


  // alloc data2

  std::vector<element_t> data2_vec(matrix_t::s_num_rows*matrix_t::s_num_columns);
  RAJA::View<element_t, RAJA::Layout<2>> data2_h(data2_vec.data(), matrix_t::s_num_columns, matrix_t::s_num_rows);

  element_t *data2_ptr = tensor_malloc(data2_vec);
  RAJA::View<element_t, RAJA::Layout<2>> data2_d(data2_ptr, matrix_t::s_num_columns, matrix_t::s_num_rows);


  // Fill data
  for(camp::idx_t i = 0;i < 2*matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < 2*matrix_t::s_num_columns; ++ j){
      data1_h(j,i) = 2*i*matrix_t::s_num_columns+j;
    }
  }

  tensor_copy_to_device(data1_ptr, data1_vec);


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

  tensor_copy_to_host(data2_vec, data2_ptr);


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

      tensor_copy_to_device(data2_ptr, data2_vec);


      //
      // Do Operation: Full load
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

      tensor_copy_to_host(data2_vec, data2_ptr);


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
  tensor_free(data1_ptr);
  tensor_free(data2_ptr);
}






#if 0


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
    m.store_packed(&data1[0][0], matrix_t::s_num_columns, 1);
  }
  else{
    printf("store_strided\n");
    m.store_strided(&data1[0][0], matrix_t::s_num_columns, 1);
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



  // Row-Major data sub-slice
  element_t data1sub[matrix_t::s_num_rows*2][matrix_t::s_num_columns*2];

  if(matrix_t::layout_type::is_row_major()){
    printf("store_packed\n");
    m.store_packed(&data1sub[0][0], matrix_t::s_num_columns*2, 1);
  }
  else{
    printf("store_strided\n");
    m.store_strided(&data1sub[0][0], matrix_t::s_num_columns*2, 1);
  }

  // Check contents
  printf("data1sub:\n");
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      printf("%lf ", data1sub[i][j]);
    }
    printf("\n");
  }
  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data1sub[i][j]);
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



  // Column-Major data sub-slice
  element_t data2sub[matrix_t::s_num_columns*2][matrix_t::s_num_rows*2];

  if(matrix_t::layout_type::is_column_major()){
    printf("store_packed\n");
    m.store_packed(&data2sub[0][0], 1, matrix_t::s_num_rows*2);
  }
  else{
    printf("store_strided\n");
    m.store_strided(&data2sub[0][0], 1, matrix_t::s_num_rows*2);
  }

  // Check contents
  printf("data2sub:\n");
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      printf("%lf ", data2sub[j][i]);
    }
    printf("\n");
  }
  // Check contents
  for(camp::idx_t i = 0;i < matrix_t::s_num_rows; ++ i){
    for(camp::idx_t j = 0;j < matrix_t::s_num_columns; ++ j){
      ASSERT_SCALAR_EQ(m.get(i,j), data2sub[j][i]);
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
//#endif

TYPED_TEST_P(MatrixTest, MatrixVector)
{

  using matrix_t = TypeParam;
  using element_t = typename matrix_t::element_type;
  using column_vector_t = typename matrix_t::column_vector_type;
  using row_vector_t = typename matrix_t::row_vector_type;
  static constexpr camp::idx_t num_rows = matrix_t::s_num_rows;
  static constexpr camp::idx_t num_columns = matrix_t::s_num_columns;

  // initialize a matrix and vector
  matrix_t m;
  for(camp::idx_t j = 0;j < num_columns; ++ j){
    for(camp::idx_t i = 0;i < num_rows; ++ i){
      m.set(element_t(NO_OPT_ZERO + i*num_columns + j + 1), i,j);
    }
  }


  {
    row_vector_t v;
    for(camp::idx_t i = 0;i < num_columns; ++ i){
      v.set(NO_OPT_ZERO + i + 1, i);
    }


    // matrix vector product
    // note mv is not necessarily the same type as v
    auto mv = m.right_multiply_vector(v);

//    printf("m: %s", m.to_string().c_str());
//    printf("v: %s", v.to_string().c_str());
//    printf("mv: %s", mv.to_string().c_str());

    // check result
    for(camp::idx_t i = 0;i < num_rows; ++ i){
      element_t expected(0);

      for(camp::idx_t j = 0;j < num_columns; ++ j){
        expected += m.get(i,j)*v.get(j);
      }

//      printf("mv: i=%d, val=%lf, expected=%lf\n", (int)i, (double)mv.get(i), (double)expected);

      ASSERT_SCALAR_EQ(mv.get(i), expected);
    }
  }

  {

    column_vector_t v;
    for(camp::idx_t j = 0;j < num_rows; ++ j){
      v.set(NO_OPT_ZERO + j + 1, j);
    }

    // matrix vector product
    auto vm = m.left_multiply_vector(v);

//    printf("vm: %s", vm.to_string().c_str());

    // check result
    for(camp::idx_t j = 0;j < num_columns; ++ j){
      element_t expected(0);

      for(camp::idx_t i = 0;i < num_rows; ++ i){
        expected += m.get(i,j)*v.get(i);
      }

//      printf("vm: j=%d, val=%lf, expected=%lf\n", (int)j, (double)vm.get(j), (double)expected);


      ASSERT_SCALAR_EQ(vm.get(j), expected);
    }
  }
}

//#endif
//#if 0
TYPED_TEST_P(MatrixTest, MatrixMatrix)
{

  static constexpr camp::idx_t N = TypeParam::s_num_rows;
  static constexpr camp::idx_t M = TypeParam::s_num_columns;
  using element_t = typename TypeParam::element_type;
  using layout_t = typename TypeParam::layout_type;

  using A_t = TypeParam;
  using B_t = RAJA::RectMatrixRegister<element_t, layout_t, M, N>;
  using C_t = RAJA::RectMatrixRegister<element_t, layout_t, N, N>;



  // initialize two matrices
  A_t A;
  A.clear();

  for(camp::idx_t j = 0;j < M; ++ j){
    for(camp::idx_t i = 0;i < N; ++ i){
      A.set(element_t(NO_OPT_ZERO + i*M+j), i,j);
    }
  }

  B_t B;
  B.clear();
  for(camp::idx_t j = 0;j < N; ++ j){
    for(camp::idx_t i = 0;i < M; ++ i){
      B.set(element_t(NO_OPT_ZERO + i*N+j), i,j);
    }
  }

//  printf("A:\n%s\n", A.to_string().c_str());
//  printf("B:\n%s\n", B.to_string().c_str());

  // matrix matrix product
  C_t C = A.matrix_multiply(B);

//  printf("C:\n%s\n", C.to_string().c_str());


  // check result
  for(camp::idx_t i = 0;i < N; ++ i){

    for(camp::idx_t j = 0;j < N; ++ j){

      // dot product to compute C(i,j)
      element_t expected(0);
      for(camp::idx_t k = 0;k < M; ++ k){
        expected += A.get(i, k) * B.get(k,j);
      }

      ASSERT_SCALAR_EQ(C.get(i,j), expected);
    }
  }


}

//#if 0

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
//                                          MatrixGetSet,
    MatrixLoad_RowMajor,
    MatrixLoad_ColMajor
//                                          MatrixStore,
//                                          MatrixViewLoad,
//                                          MatrixViewStore,
//                                          MatrixVector,
//                                          MatrixMatrix
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






