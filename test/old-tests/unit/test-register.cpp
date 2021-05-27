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

//    RAJA::VectorRegister<double, RAJA::avx2_register, 8>,
    RAJA::VectorRegister<double, RAJA::avx_register, 5>,
    RAJA::VectorRegister<double, RAJA::avx_register, 6>,
    RAJA::VectorRegister<double, RAJA::avx_register, 7>,
    RAJA::VectorRegister<double, RAJA::avx_register, 8>,
    RAJA::VectorRegister<double, RAJA::avx_register, 9>
//    RAJA::VectorRegister<float>,
//    RAJA::VectorRegister<int>,
//    RAJA::VectorRegister<long>,

//#ifdef __AVX__
//    RAJA::Register<double, RAJA::avx_register>,
//    RAJA::Register<float, RAJA::avx_register>,
//    RAJA::Register<int, RAJA::avx_register>,
//    RAJA::Register<long, RAJA::avx_register>,
//#endif
//
//#ifdef __AVX2__
//    RAJA::Register<double, RAJA::avx2_register>,
//    RAJA::Register<float, RAJA::avx2_register>,
//    RAJA::Register<int, RAJA::avx2_register>,
//    RAJA::Register<long, RAJA::avx2_register>,
//#endif
//
//#ifdef __AVX512__
//    RAJA::Register<double, RAJA::avx512_register>,
//    RAJA::Register<float, RAJA::avx512_register>,
//    RAJA::Register<int, RAJA::avx512_register>,
//    RAJA::Register<long, RAJA::avx512_register>,
//#endif
//
//    // scalar_register is supported on all platforms
//    RAJA::Register<double, RAJA::scalar_register>,
//    RAJA::Register<float, RAJA::scalar_register>,
//    RAJA::Register<int, RAJA::scalar_register>,
//    RAJA::Register<long, RAJA::scalar_register>
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

TYPED_TEST_P(RegisterTest, RegisterSetGet)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem];
  register_t x;
  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
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


TYPED_TEST_P(RegisterTest, RegisterLoad)
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

TYPED_TEST_P(RegisterTest, Gather)
{

  using register_t = TypeParam;

  using int_vector_type = typename register_t::int_vector_type;

  using element_t = typename register_t::element_type;
  static constexpr camp::idx_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*num_elem];
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
    A[i] = 3*i+13;
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
  }

  // create an index vector to point at sub elements of A
  int_vector_type idx;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
    idx.set(j*j, i);
//    printf("idx[%d]=%d\n", (int)i, (int)(j*j));
  }

  // Gather elements from A into a register using the idx offsets
  register_t x;
  x.gather(&A[0], idx);

  // check
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
//    printf("i=%d, j=%d, A[%d]=%d, x.get(i)=%d\n",
//        (int)i, (int)j, (int)(j*j), (int)A[j*j], (int)x.get(i));
    ASSERT_SCALAR_EQ(A[j*j], x.get(i));
  }


  // Gather all but one elements from A into a register using the idx offsets
  register_t y;
  y.gather_n(&A[0], idx, num_elem-1);

  // check
  for(camp::idx_t i = 0;i < num_elem-1;++ i){
    int j = num_elem-1-i;
    ASSERT_SCALAR_EQ(A[j*j], y.get(i));
  }
  ASSERT_SCALAR_EQ(0, y.get(num_elem-1));


}

TYPED_TEST_P(RegisterTest, Scatter)
{

  using register_t = TypeParam;

  using int_vector_type = typename register_t::int_vector_type;

  using element_t = typename register_t::element_type;
  static constexpr camp::idx_t num_elem = register_t::s_num_elem;

  element_t A[num_elem*num_elem];
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
    A[i] = 0;
  }

  // create an index vector to point at sub elements of A
  int_vector_type idx;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    int j = num_elem-1-i;
    idx.set(j*j, i);
  }

  // Create a vector of values
  register_t x;
  for(camp::idx_t i = 0;i < num_elem;++ i){
    x.set(i+1, i);
  }

  // Scatter the values of x into A[] using idx as the offsets
  x.scatter(&A[0], idx);

  // check
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
    // if the index i is in idx, check that A contains the right value
    for(camp::idx_t j = 0;j < num_elem;++ j){
      if(idx.get(j) == i){
        // check
        ASSERT_SCALAR_EQ(A[i], element_t(j+1));
        // and set to zero (for the next assert, and to clear for next test)
        A[i] = 0;
      }
    }
    // otherwise A should contain zero
    ASSERT_SCALAR_EQ(A[i], element_t(0));
  }


  // Scatter all but one of the values of x into A[] using idx as the offsets
  x.scatter_n(&A[0], idx, num_elem-1);

  // check
  for(camp::idx_t i = 0;i < num_elem*num_elem;++ i){
//    printf("A[%d]=%d\n", (int)i, (int)A[i]);
    // if the index i is in idx, check that A contains the right value
    for(camp::idx_t j = 0;j < num_elem-1;++ j){
      if(idx.get(j) == i){
        // check
        ASSERT_SCALAR_EQ(A[i], element_t(j+1));
        // and set to zero (for the next assert, and to clear for next test)
        A[i] = 0;
      }
    }
    // otherwise A should contain zero
    ASSERT_SCALAR_EQ(A[i], element_t(0));
  }


}


TYPED_TEST_P(RegisterTest, RegisterAdd)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
    y.set(B[i], i);
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





TYPED_TEST_P(RegisterTest, RegisterSubtract)
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
    x.set(A[i], i);
    y.set(B[i], i);
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

TYPED_TEST_P(RegisterTest, RegisterMultiply)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0);
    B[i] = (element_t)(NO_OPT_RAND*1000.0);
    x.set(A[i], i);
    y.set(B[i], i);
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

TYPED_TEST_P(RegisterTest, RegisterDivide)
{

  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  element_t A[num_elem], B[num_elem];
  register_t x, y;

  for(size_t i = 0;i < num_elem; ++ i){
    A[i] = (element_t)(NO_OPT_RAND*1000.0)+1.0;
    B[i] = (element_t)(NO_OPT_RAND*1000.0)+1.0;
    x.set(A[i], i);
    y.set(B[i], i);
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

TYPED_TEST_P(RegisterTest, RegisterDotProduct)
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
    x.set(A[i], i);
    y.set(B[i], i);
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
    x.set(A[i], i);
    y.set(B[i], i);
    z.set(C[i], i);
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
    x.set(A[i], i);
    y.set(B[i], i);
    z.set(C[i], i);
    expected[i] = A[i]*B[i]-C[i];
  }

  result = x.multiply_subtract(y,z);

  for(size_t i = 0;i < num_elem; ++ i){
    ASSERT_SCALAR_EQ(result.get(i), expected[i]);
  }

}

TYPED_TEST_P(RegisterTest, RegisterMax)
{
  using register_t = TypeParam;

  using element_t = typename register_t::element_type;
  static constexpr size_t num_elem = register_t::s_num_elem;

  for(int iter = 0;iter < 100;++ iter){
    element_t A[num_elem], B[num_elem];
    register_t x, y;

    for(size_t i = 0;i < num_elem; ++ i){
      A[i] = -(element_t)(NO_OPT_RAND*1000.0);
      B[i] = -(element_t)(NO_OPT_RAND*1000.0);
      x.set(A[i], i);
      y.set(B[i], i);
    }

    // Check vector reduction
    element_t expected = A[0];
    for(size_t i = 1;i < num_elem;++ i){
      expected = expected > A[i] ? expected : A[i];
    }

//    printf("X=%s", x.to_string().c_str());

    ASSERT_SCALAR_EQ(x.max(), expected);


    // Check element-wise
    register_t z = x.vmax(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_SCALAR_EQ(z.get(i), std::max<element_t>(A[i], B[i]));
    }


  }
}

TYPED_TEST_P(RegisterTest, RegisterMin)
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
      x.set(A[i], i);
      y.set(B[i], i);
    }

    // Check vector reduction
    element_t expected = A[0];
    for(size_t i = 1;i < num_elem;++ i){
      expected = expected < A[i] ? expected : A[i];
    }

//    printf("X=%s", x.to_string().c_str());


    ASSERT_SCALAR_EQ(x.min(), expected);

    // Check element-wise
    register_t z = x.vmin(y);
    for(size_t i = 1;i < num_elem;++ i){
      ASSERT_SCALAR_EQ(z.get(i), std::min<element_t>(A[i], B[i]));
    }

  }
}


#endif



//REGISTER_TYPED_TEST_SUITE_P(RegisterTest, RegisterSubtract);

REGISTER_TYPED_TEST_SUITE_P(RegisterTest,
                                       RegisterSetGet,
                                       RegisterLoad,
                                       Gather,
                                       Scatter,
                                       RegisterAdd,
                                       RegisterSubtract,
                                       RegisterMultiply,
                                       RegisterDivide,
                                       RegisterDotProduct,
                                       VectorFMA,
                                       VectorFMS,
                                       RegisterMax,
                                       RegisterMin);

INSTANTIATE_TYPED_TEST_SUITE_P(SIMD, RegisterTest, RegisterTestTypes);


#if defined(RAJA_ENABLE_CUDA)


GPU_TEST(RegisterTestCuda, CudaWarpRegister)
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


  using vector_t = RAJA::Register<double, RAJA::cuda_warp_register>;

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_loop,
      RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
          RAJA::statement::Lambda<0>
            >
          >
        >
       >;


  RAJA::kernel<Pol>(

      RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N*32)),

      [=] __device__(int i){
        // limit i to increments of 32
        i -= i & 0x1F;

        // load vector
        vector_t v;
        v.load_packed(data+i);

        // scale by 2
        v = v.scale(2.0);

        // store in result
        v.store_packed(result+i);

      });


  cudaErrchk(cudaDeviceSynchronize());


  for(int i = 0;i < N;++ i){
    ASSERT_SCALAR_EQ(data[i]*2.0, result[i]);
  }



}



#endif // RAJA_ENABLE_CUDA

