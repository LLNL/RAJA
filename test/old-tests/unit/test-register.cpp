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
#include <stdlib.h>

    using RegisterTestTypes = ::testing::Types<
#ifdef __AVX__
       RAJA::Register<RAJA::vector_avx_register, double, 2>,
       RAJA::Register<RAJA::vector_avx_register, double, 3>,
       RAJA::Register<RAJA::vector_avx_register, double, 4>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,1>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,2>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,3>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 27>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 4>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx_register, double,4>, 8>,
#endif

#ifdef __AVX2__
       RAJA::Register<RAJA::vector_avx2_register, double, 2>,
       RAJA::Register<RAJA::vector_avx2_register, double, 3>,
       RAJA::Register<RAJA::vector_avx2_register, double, 4>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,1>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,2>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,3>, 27>,
       RAJA::FixedVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 27>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 4>,
       RAJA::StreamVectorExt<RAJA::Register<RAJA::vector_avx2_register, double,4>, 8>,
#endif
       RAJA::Register<RAJA::vector_scalar_register, int, 1>,
       RAJA::Register<RAJA::vector_scalar_register, float, 1>,
       RAJA::Register<RAJA::vector_scalar_register, double, 1>,

       // Test automatically wrapped types to make things easier for users
       RAJA::StreamVector<double>,
       RAJA::StreamVector<double, 2>,
       RAJA::FixedVector<double, 1>,
       RAJA::FixedVector<double, 2>,
       RAJA::FixedVector<double, 4>,
       RAJA::FixedVector<double, 8>,
       RAJA::FixedVector<double, 16>>;

//using RegisterTestTypes = ::testing::Types<RAJA::Register<RAJA::vector_scalar_register, double, 1>>;

template <typename NestedPolicy>
class RegisterTest : public ::testing::Test
{
protected:

  RegisterTest() = default;
  virtual ~RegisterTest() = default;

  virtual void SetUp()
  {
  }

  virtual void TearDown()
  {
  }
};
TYPED_TEST_SUITE_P(RegisterTest);


/*
 * We are using ((double)rand()/RAND_MAX) for input values so the compiler cannot do fancy
 * things, like constexpr out all of the intrinsics.
 */

TYPED_TEST_P(RegisterTest, VectorRegisterSetGet)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
  }

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(x[i], A[i]);
  }

}


TYPED_TEST_P(RegisterTest, VectorRegisterLoad)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*2];
  for(size_t i = 0;i < num_elem*2; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
  }


  // load stride-1 from pointer
  register_t x;
  x.load(A);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(x[i], A[i]);
  }

  // load stride-2from pointer
  register_t y;
  y.load(A, 2);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(y[i], A[i*2]);
  }
}



TYPED_TEST_P(RegisterTest, VectorRegisterAdd)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    B[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x+y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] + B[i]);
  }

  register_t z2 = x;
  z2 += y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] + B[i]);
  }

}





TYPED_TEST_P(RegisterTest, VectorRegisterSubtract)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x;
  register_t y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    B[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x-y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_EQ(z[i], A[i] - B[i]);
  }

}

TYPED_TEST_P(RegisterTest, VectorRegisterMultiply)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    B[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x*y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] * B[i]);
  }

  register_t z2 = x;
  z2 *= y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] * B[i]);
  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterDivide)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    B[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0)+1.0;
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  register_t z = x/y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z[i], A[i] / B[i]);
  }

  register_t z2 = x;
  z2 /= y;

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(z2[i], A[i] / B[i]);
  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterDotProduct)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  element_t expected = 0.0;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    B[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
    expected += A[i]*B[i];
  }

  ASSERT_DOUBLE_EQ(x.dot(y), expected);

}

TYPED_TEST_P(RegisterTest, VectorRegisterMax)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
  }

  element_t expected = A[0];
  for(size_t i = 1;i < num_elem;++ i){
    expected = expected > A[i] ? expected : A[i];
  }

  ASSERT_DOUBLE_EQ(x.max(), expected);

}

TYPED_TEST_P(RegisterTest, VectorRegisterMin)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(((double)rand()/RAND_MAX)*1000.0);
    x.set(i, A[i]);
  }

  element_t expected = A[0];
  for(size_t i = 1;i < num_elem;++ i){
    expected = expected < A[i] ? expected : A[i];
  }

  ASSERT_DOUBLE_EQ(x.min(), expected);

}

//REGISTER_TYPED_TEST_SUITE_P(RegisterTest, VectorRegisterSubtract);

REGISTER_TYPED_TEST_SUITE_P(RegisterTest, VectorRegisterSetGet,
                                       VectorRegisterLoad,
                                       VectorRegisterAdd,
                                       VectorRegisterSubtract,
                                       VectorRegisterMultiply,
                                       VectorRegisterDivide,
                                       VectorRegisterDotProduct,
                                       VectorRegisterMax,
                                       VectorRegisterMin);

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, RegisterTest, RegisterTestTypes);


#if defined(RAJA_ENABLE_CUDA)


GPU_TEST(RegisterTestCuda, CudaWarp32)
{
  using namespace RAJA::statement;

  using element_t = double;
  size_t N = 20;

  element_t *data = nullptr;

  cudaErrchk(cudaMallocManaged(&data,
                    sizeof(element_t) * N*32,
                    cudaMemAttachGlobal));

  element_t *result = nullptr;

  cudaErrchk(cudaMallocManaged(&result,
                    sizeof(element_t) * N,
                    cudaMemAttachGlobal));

  cudaErrchk(cudaDeviceSynchronize());

  for(int i = 0;i < N*32;++ i){
    data[i] = i; //1000*((double)rand()/RAND_MAX);
  }

  for(int i = 0;i < N;++ i){
    result[i] = 0.0;
  }

  cudaErrchk(cudaDeviceSynchronize());


  using register_t = RAJA::Register<RAJA::vector_cuda_warp_register<5>, element_t, 32>;

  using vector_t = RAJA::FixedVectorExt<register_t, 32>;

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<32>, RAJA::cuda_block_x_loop,
      RAJA::statement::For<0, RAJA::cuda_warp_vector_loop<vector_t>,
          RAJA::statement::Lambda<0>
            >
          >
        >
       >;

  auto data_view = RAJA::make_view<int>(data);

  RAJA::kernel<Pol>(

      RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N*32)),

      [=] __device__(RAJA::VectorIndex<int, vector_t> i){
        auto value = data_view(i);

        element_t s = value.sum();

        if(vector_t::is_root()){
           result[(*i)>>5] = s;
        }

      });


  cudaErrchk(cudaDeviceSynchronize());


  for(int i = 0;i < N;++ i){
    element_t expected = data[i*32];
    for(int j = 1;j <32;++ j){
      expected += data[i*32+j];
    }
    ASSERT_DOUBLE_EQ(expected, result[i]);
  }



}


GPU_TEST(RegisterTestCuda, CudaWarp16)
{
  using namespace RAJA::statement;

  using element_t = double;
  size_t N = 2;

  element_t *data = nullptr;

  cudaErrchk(cudaMallocManaged(&data,
                    sizeof(element_t) * N*32,
                    cudaMemAttachGlobal));

  element_t *result = nullptr;

  cudaErrchk(cudaMallocManaged(&result,
                    sizeof(element_t) * N*2,
                    cudaMemAttachGlobal));

  cudaErrchk(cudaDeviceSynchronize());

  for(int i = 0;i < N*32;++ i){
    data[i] = 1000*((double)rand()/RAND_MAX);
  }

  for(int i = 0;i < N*2;++ i){
    result[i] = 0.0;
  }

  cudaErrchk(cudaDeviceSynchronize());


  using register_t = RAJA::Register<RAJA::vector_cuda_warp_register<4>, element_t, 16>;


  using Pol = RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<32>, RAJA::cuda_block_x_loop,
      RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
          RAJA::statement::Lambda<0>
            >
          >
        >
       >;

  RAJA::kernel<Pol>(

      RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N*32)),

      [=] __device__(int i0){
        int i = (i0>>4)<<4;
        register_t value;
        value.load(data + i);
        element_t s = value.sum();
        if(register_t::is_root()){
           result[i0>>4] = s;
        }
      });


  cudaErrchk(cudaDeviceSynchronize());


  for(int i = 0;i < N*2;++ i){
    element_t expected = data[i*16];
    for(int j = 1;j <16;++ j){
      expected += data[i*16+j];
    }
    ASSERT_DOUBLE_EQ(expected, result[i]);
  }



}


#endif // RAJA_ENABLE_CUDA

