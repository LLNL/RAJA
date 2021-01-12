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

#ifdef __AVX__
    RAJA::VectorRegister<double, RAJA::avx_register>,
    RAJA::VectorRegister<float, RAJA::avx_register>,
    RAJA::VectorRegister<int, RAJA::avx_register>,
    RAJA::VectorRegister<long, RAJA::avx_register>,
#endif

#ifdef __AVX2__
    RAJA::VectorRegister<double, RAJA::avx2_register>,
    RAJA::VectorRegister<float, RAJA::avx2_register>,
    RAJA::VectorRegister<int, RAJA::avx2_register>,
    RAJA::VectorRegister<long, RAJA::avx2_register>,
#endif

#ifdef __AVX512__
    RAJA::VectorRegister<double, RAJA::avx512_register>,
    RAJA::VectorRegister<float, RAJA::avx512_register>,
    RAJA::VectorRegister<int, RAJA::avx512_register>,
    RAJA::VectorRegister<long, RAJA::avx512_register>,
#endif

    // scalar_register is supported on all platforms
    RAJA::VectorRegister<double, RAJA::scalar_register>,
    RAJA::VectorRegister<float, RAJA::scalar_register>,
    RAJA::VectorRegister<int, RAJA::scalar_register>,
    RAJA::VectorRegister<long, RAJA::scalar_register>
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

#if 1
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
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
  }

  // test copy construction
  register_t cc(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(cc.get(i), A[i]);
  }

  // test explicit copy
  register_t ce(0);
  ce.copy(x);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ce.get(i), A[i]);
  }

  // test assignment
  register_t ca(0);
  ca = cc;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ca.get(i), A[i]);
  }

  // test scalar construction (broadcast)
  register_t bc((element_t)5);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(bc.get(i), 5.0);
  }

  // test scalar assignment (broadcast)
  register_t ba((element_t)0);
  ba = (element_t)13.0;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(ba.get(i), 13.0);
  }

  // test explicit broadcast
  register_t be((element_t)0);
  be.broadcast((element_t)13.0);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(be.get(i), 13.0);
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
  x.load_packed(A);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(x.get(i), A[i]);
  }


  // load n stride-1 from pointer
  if(num_elem > 1){
    x.load_packed_n(A, num_elem-1);

    // check first n-1 values
    for(size_t i = 0;i+1 < num_elem; ++ i){
      ASSERT_SCALAR_EQ(x.get(i), A[i]);
    }

    // last value should be cleared to zero
    ASSERT_SCALAR_EQ(x.get(num_elem-1), 0);
  }

  // load stride-2 from pointer
  register_t y;
  y.load_strided(A, 2);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(y.get(i), A[i*2]);
  }

  // load n stride-2 from pointer
  if(num_elem > 1){
    y.load_strided_n(A, 2, num_elem-1);

    // check first n-1 values
    for(size_t i = 0;i+1 < num_elem; ++ i){
      ASSERT_SCALAR_EQ(y.get(i), A[i*2]);
    }

    // last value should be cleared to zero
    ASSERT_SCALAR_EQ(y.get(num_elem-1), 0);
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
    ASSERT_SCALAR_EQ(op_add.get(i), A[i] + B[i]);
  }

  // operator +=
  register_t op_pluseq = x;
  op_pluseq += y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_pluseq.get(i), A[i] + B[i]);
  }

  // function add
  register_t func_add = x.add(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_add.get(i), A[i] + B[i]);
  }

  // operator + scalar
  register_t op_add_s1 = x + element_t(1);
  register_t op_add_s2 = element_t(1) + x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_add_s1.get(i), A[i] + element_t(1));
    ASSERT_SCALAR_EQ(op_add_s2.get(i), element_t(1) + A[i]);
  }

  // operator += scalar
  register_t op_pluseq_s = x;
  op_pluseq_s += element_t(1);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_pluseq_s.get(i), A[i] + element_t(1));
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
    ASSERT_SCALAR_EQ(op_sub.get(i), A[i] - B[i]);
  }

  // operator -=
  register_t op_subeq = x;
  op_subeq -= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_subeq.get(i), A[i] - B[i]);
  }

  // function subtract
  register_t func_sub = x.subtract(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_sub.get(i), A[i] - B[i]);
  }

  // operator - scalar
  register_t op_sub_s1 = x - element_t(1);
  register_t op_sub_s2 = element_t(1) - x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_sub_s1.get(i), A[i] - element_t(1));
    ASSERT_SCALAR_EQ(op_sub_s2.get(i), element_t(1) - A[i]);
  }

  // operator -= scalar
  register_t op_subeq_s = x;
  op_subeq_s -= element_t(1);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_subeq_s.get(i), A[i] - element_t(1));
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
    ASSERT_SCALAR_EQ(op_mul.get(i), (A[i] * B[i]));
  }

  // operator *=
  register_t op_muleq = x;
  op_muleq *= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_muleq.get(i), A[i] * B[i]);
  }

  // function multiply
  register_t func_mul = x.multiply(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_mul.get(i), A[i] * B[i]);
  }

  // operator * scalar
  register_t op_mul_s1 = x * element_t(2);
  register_t op_mul_s2 = element_t(2) * x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_mul_s1.get(i), A[i] * element_t(2));
    ASSERT_SCALAR_EQ(op_mul_s2.get(i), element_t(2) * A[i]);
  }

  // operator *= scalar
  register_t op_muleq_s = x;
  op_muleq_s *= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_muleq_s.get(i), A[i] * element_t(2));
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
    ASSERT_SCALAR_EQ(op_div.get(i), A[i] / B[i]);
  }

  // operator /=
  register_t op_diveq = x;
  op_diveq /= y;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_diveq.get(i), A[i] / B[i]);
  }

  // function divide
  register_t func_div = x.divide(y);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(func_div.get(i), A[i] / B[i]);
  }


  // operator / scalar
  register_t op_div_s1 = x / element_t(2);
  register_t op_div_s2 = element_t(2) / x;
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_div_s1.get(i), A[i] / element_t(2));
    ASSERT_SCALAR_EQ(op_div_s2.get(i), element_t(2) / A[i]);
  }

  // operator /= scalar
  register_t op_diveq_s = x;
  op_diveq_s /= element_t(2);
  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(op_diveq_s.get(i), A[i] / element_t(2));
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

  ASSERT_SCALAR_EQ(x.dot(y), expected);

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

  result = x.multiply_add(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(result.get(i), expected[i]);
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

  result = x.multiply_subtract(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(result.get(i), expected[i]);
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

    ASSERT_SCALAR_EQ(x.max(), expected);


    // Check element-wise
    register_t z = x.vmax(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_SCALAR_EQ(z.get(i), std::max<element_t>(A[i], B[i]));
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

    ASSERT_SCALAR_EQ(x.min(), expected);

    // Check element-wise
    register_t z = x.vmin(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_SCALAR_EQ(z.get(i), std::min<element_t>(A[i], B[i]));
    }

  }
}


#endif



//REGISTER_TYPED_TEST_SUITE_P(RegisterTest, VectorRegisterSubtract);

REGISTER_TYPED_TEST_SUITE_P(RegisterTest,
    VectorRegisterSetGet,
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


  //using register_t = RAJA::VectorRegister<RAJA::cuda_warp_register<5>, element_t, 32>;

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
    ASSERT_SCALAR_EQ(expected, result[i]);
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


  using register_t = RAJA::VectorRegister<RAJA::cuda_warp_register<4>, element_t>;


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
        value.load_packed(data + i);
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
    ASSERT_SCALAR_EQ(expected, result[i]);
  }



}


#endif // RAJA_ENABLE_CUDA

