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
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,8, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4, RAJA::cuda_warp_register>,
    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,8, RAJA::cuda_warp_register>,
#endif

//    // These tests use the platform default SIMD architecture
//    RAJA::SquareMatrixRegister<double, RAJA::ColMajorLayout>
//    RAJA::SquareMatrixRegister<double, RAJA::RowMajorLayout>,

//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 8,2>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,4>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 4,8>,
//    RAJA::RectMatrixRegister<double, RAJA::ColMajorLayout, 2,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,4>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 8,2>,
//    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,4>,
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 4,8>,
    RAJA::RectMatrixRegister<double, RAJA::RowMajorLayout, 2,4>
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

    static constexpr bool is_device = false;
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

    static constexpr bool is_device = true;
};
#endif


template<typename POL, typename BODY>
void tensor_do(BODY const &body){
  TensorTestHelper<POL>::exec(body);
}



#ifdef RAJA_ENABLE_CUDA

template<typename POL, typename T>
T* tensor_malloc(size_t len){
  if(TensorTestHelper<POL>::is_device){
    T *ptr;

    cudaErrchk(cudaMalloc(&ptr, len*sizeof(T)));

    return ptr;
  }
  else{
    return new T[len];
  }
}

template<typename POL, typename T>
void tensor_free(T *ptr){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaFree(ptr));
  }
  else{
    delete[] ptr;
  }
}

template<typename POL, typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T), cudaMemcpyHostToDevice));
  }
  else{
    memcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T));
  }
}

template<typename POL, typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  if(TensorTestHelper<POL>::is_device){
    cudaErrchk(cudaMemcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T), cudaMemcpyDeviceToHost));
  }
  else{
    memcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T));
  }
}

#else

template<typename POL, typename T>
T* tensor_malloc(size_t len){
  return new T[len];
}

template<typename POL, typename T>
void tensor_free(T *ptr){
  delete[] ptr;
}

template<typename POL, typename T>
void tensor_copy_to_device(T *d_ptr, std::vector<T> const &h_vec){
  memcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(T));
}

template<typename POL, typename T>
void tensor_copy_to_host(std::vector<T> &h_vec, T const *d_ptr){
  memcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(T));
}

#endif



// Sugar to make things cleaner
template<typename POL, typename T>
T* tensor_malloc(std::vector<T> const &vec){
  return tensor_malloc<POL,T>(vec.size());
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

  static constexpr camp::idx_t N = matrix_t::s_num_rows*matrix_t::s_num_columns*2;

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

  static constexpr camp::idx_t N = matrix_t::s_num_rows*matrix_t::s_num_columns*2;

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

  static constexpr camp::idx_t N = matrix_t::s_num_rows*matrix_t::s_num_columns*2;

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


#if 0


TYPED_TEST_P(MatrixTest, ET_MatrixVectorMultiply)
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


TYPED_TEST_P(MatrixTest, ET_MatrixMatrixMultiply)
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


TYPED_TEST_P(MatrixTest, ET_MatrixMatrixMultiplyAdd)
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
                                          ET_Divide
//                                        ET_MatrixVectorMultiply,
//                                        ET_MatrixMatrixMultiply,
//                                        ET_MatrixMatrixMultiplyAdd,
//                                        ET_TransposeNegate
                                        );

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, MatrixTest, MatrixTestTypes);






