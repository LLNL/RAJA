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

using RegisterTestTypes = ::testing::Types<
#ifdef RAJA_ALTIVEC
    RAJA::Register<RAJA::altivec_register, double>,
    RAJA::Register<RAJA::altivec_register, float>,
    RAJA::Register<RAJA::altivec_register, int>,
    RAJA::Register<RAJA::altivec_register, long>,
#endif

#ifdef __AVX__
    RAJA::Register<RAJA::avx_register, double>,
    RAJA::Register<RAJA::avx_register, float>,
    RAJA::Register<RAJA::avx_register, int>,
    RAJA::Register<RAJA::avx_register, long>,
#endif

#ifdef __AVX2__
    RAJA::Register<RAJA::avx2_register, double>,
    RAJA::Register<RAJA::avx2_register, float>,
    RAJA::Register<RAJA::avx2_register, int>,
    RAJA::Register<RAJA::avx2_register, long>,
#endif

    // scalar_register is supported on all platforms
    RAJA::Register<RAJA::scalar_register, int>,
    RAJA::Register<RAJA::scalar_register, long>,
    RAJA::Register<RAJA::scalar_register, float>,
    RAJA::Register<RAJA::scalar_register, double>,

//    // Test automatically wrapped types to make things easier for users
    RAJA::StreamVector<int>,
    RAJA::StreamVector<int, 2>,
    RAJA::StreamVector<long>,
    RAJA::StreamVector<long, 2>,
    RAJA::StreamVector<float>,
    RAJA::StreamVector<float, 2>,
    RAJA::StreamVector<double>,
    RAJA::StreamVector<double, 2>,
    RAJA::FixedVector<double, 1>,
    RAJA::FixedVector<double, 2>,
    RAJA::FixedVector<double, 3>,
    RAJA::FixedVector<double, 4>,
    RAJA::FixedVector<double, 5>,
    RAJA::FixedVector<long, 1>,
    RAJA::FixedVector<long, 2>,
    RAJA::FixedVector<long, 3>,
    RAJA::FixedVector<long, 4>,
    RAJA::FixedVector<long, 5>

  >;


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
 * We are using NO_OPT_RAND for input values so the compiler cannot do fancy
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
  }

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(x[i], A[i]);
    ASSERT_DOUBLE_EQ(x.get(i), A[i]);
  }

  // test copy construction
  register_t cc(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(cc[i], A[i]);
  }

  // test explicit copy
  register_t ce(0);
  ce.copy(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(ce[i], A[i]);
  }

  // test assignment
  register_t ca(0);
  ca = cc;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(ca[i], A[i]);
  }

  // test scalar construction (broadcast)
  register_t bc((element_t)5);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(bc[i], 5.0);
  }

  // test scalar assignment (broadcast)
  register_t ba((element_t)0);
  ba = (element_t)13.0;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(ba[i], 13.0);
  }

  // test explicit broadcast
  register_t be((element_t)0);
  be.broadcast((element_t)13.0);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(be[i], 13.0);
  }
}


TYPED_TEST_P(RegisterTest, VectorRegisterLoad)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*2];
  for(size_t i = 0;i < num_elem*2; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  // operator +
  register_t op_add = x+y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_add[i], A[i] + B[i]);
  }

  // operator +=
  register_t op_pluseq = x;
  op_pluseq += y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_pluseq[i], A[i] + B[i]);
  }

  // function add
  register_t func_add = x.add(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(func_add[i], A[i] + B[i]);
  }

  // operator + scalar
  register_t op_add_s1 = x + element_t(1);
  register_t op_add_s2 = element_t(1) + x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_add_s1[i], A[i] + element_t(1));
    ASSERT_DOUBLE_EQ(op_add_s2[i], element_t(1) + A[i]);
  }

  // operator += scalar
  register_t op_pluseq_s = x;
  op_pluseq_s += element_t(1);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_pluseq_s[i], A[i] + element_t(1));
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  // operator -
  register_t op_sub = x-y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_sub[i], A[i] - B[i]);
  }

  // operator -=
  register_t op_subeq = x;
  op_subeq -= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_subeq[i], A[i] - B[i]);
  }

  // function subtract
  register_t func_sub = x.subtract(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(func_sub[i], A[i] - B[i]);
  }

  // operator - scalar
  register_t op_sub_s1 = x - element_t(1);
  register_t op_sub_s2 = element_t(1) - x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_sub_s1[i], A[i] - element_t(1));
    ASSERT_DOUBLE_EQ(op_sub_s2[i], element_t(1) - A[i]);
  }

  // operator -= scalar
  register_t op_subeq_s = x;
  op_subeq_s -= element_t(1);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_subeq_s[i], A[i] - element_t(1));
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  // operator *
  register_t op_mul = x*y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_mul[i], (A[i] * B[i]));
  }

  // operator *=
  register_t op_muleq = x;
  op_muleq *= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_muleq[i], A[i] * B[i]);
  }

  // function multiply
  register_t func_mul = x.multiply(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(func_mul[i], A[i] * B[i]);
  }

  // operator * scalar
  register_t op_mul_s1 = x * element_t(2);
  register_t op_mul_s2 = element_t(2) * x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_mul_s1[i], A[i] * element_t(2));
    ASSERT_DOUBLE_EQ(op_mul_s2[i], element_t(2) * A[i]);
  }

  // operator *= scalar
  register_t op_muleq_s = x;
  op_muleq_s *= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_muleq_s[i], A[i] * element_t(2));
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0)+1.0;
    x.set(i, A[i]);
    y.set(i, B[i]);
  }

  // operator /
  register_t op_div = x/y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_div[i], A[i] / B[i]);
  }

  // operator /=
  register_t op_diveq = x;
  op_diveq /= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_diveq[i], A[i] / B[i]);
  }

  // function divide
  register_t func_div = x.divide(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(func_div[i], A[i] / B[i]);
  }


  // operator / scalar
  register_t op_div_s1 = x / element_t(2);
  register_t op_div_s2 = element_t(2) / x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_div_s1[i], A[i] / element_t(2));
    ASSERT_DOUBLE_EQ(op_div_s2[i], element_t(2) / A[i]);
  }

  // operator /= scalar
  register_t op_diveq_s = x;
  op_diveq_s /= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_DOUBLE_EQ(op_diveq_s[i], A[i] / element_t(2));
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
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
    expected += A[i]*B[i];
  }

  ASSERT_FLOAT_EQ(x.dot(y), expected);

}

TYPED_TEST_P(RegisterTest, VectorFMA)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem], C[num_elem], expected[num_elem];
  register_t x, y, z, result;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
    z.set(i, C[i]);
    expected[i] = A[i]*B[i]+C[i];
  }

  result = x.fused_multiply_add(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_FLOAT_EQ(result[i], expected[i]);
  }

}


TYPED_TEST_P(RegisterTest, VectorFMS)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem], C[num_elem], expected[num_elem];
  register_t x, y, z, result;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    C[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(i, A[i]);
    y.set(i, B[i]);
    z.set(i, C[i]);
    expected[i] = A[i]*B[i]-C[i];
  }

  result = x.fused_multiply_subtract(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_FLOAT_EQ(result[i], expected[i]);
  }

}

TYPED_TEST_P(RegisterTest, VectorRegisterMax)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  for(int iter = 0;iter < 100;++ iter){
    element_t A[num_elem], B[num_elem];
    register_t x, y;

    for(size_t i = 0;i < num_elem; ++ i){
      A[i] = (element_t)(NO_OPT_RAND*1000.0);
      B[i] = (element_t)(NO_OPT_RAND*1000.0);
      x.set(i, A[i]);
      y.set(i, B[i]);
    }

    // Check vector reduction
    element_t expected = A[0];
    for(size_t i = 1;i < num_elem;++ i){
      expected = expected > A[i] ? expected : A[i];
    }

    ASSERT_DOUBLE_EQ(x.max(), expected);


    // Check element-wise
    register_t z = x.vmax(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_DOUBLE_EQ(z[i], std::max<element_t>(A[i], B[i]));
    }


  }
}

TYPED_TEST_P(RegisterTest, VectorRegisterMin)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  for(int iter = 0;iter < 100;++ iter){
    element_t A[num_elem], B[num_elem];
    register_t x, y;

    for(size_t i = 0;i < num_elem; ++ i){
      A[i] = (element_t)(NO_OPT_RAND*1000.0);
      B[i] = (element_t)(NO_OPT_RAND*1000.0);
      x.set(i, A[i]);
      y.set(i, B[i]);
    }

    // Check vector reduction
    element_t expected = A[0];
    for(size_t i = 1;i < num_elem;++ i){
      expected = expected < A[i] ? expected : A[i];
    }

    ASSERT_DOUBLE_EQ(x.min(), expected);

    // Check element-wise
    register_t z = x.vmin(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_DOUBLE_EQ(z[i], std::min<element_t>(A[i], B[i]));
    }

  }
}

//REGISTER_TYPED_TEST_SUITE_P(RegisterTest, VectorRegisterSubtract);

REGISTER_TYPED_TEST_SUITE_P(RegisterTest, VectorRegisterSetGet,
                                       VectorRegisterLoad,
                                       VectorRegisterAdd,
                                       VectorRegisterSubtract,
                                       VectorRegisterMultiply,
                                       VectorRegisterDivide,
                                       VectorRegisterDotProduct,
                                       VectorFMA,
                                       VectorFMS,
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
    data[i] = i; //1000*NO_OPT_RAND;
  }

  for(int i = 0;i < N;++ i){
    result[i] = 0.0;
  }

  cudaErrchk(cudaDeviceSynchronize());


  //using register_t = RAJA::Register<RAJA::cuda_warp_register<5>, element_t, 32>;

  using vector_t = RAJA::CudaWarpFixedVector<element_t, 32, 5>;

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_loop,
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
    data[i] = 1000*NO_OPT_RAND;
  }

  for(int i = 0;i < N*2;++ i){
    result[i] = 0.0;
  }

  cudaErrchk(cudaDeviceSynchronize());


  using register_t = RAJA::Register<RAJA::cuda_warp_register<4>, element_t>;


  using Pol = RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_loop,
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

