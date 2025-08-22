//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic min and max methods
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit-test-forone.hpp"
#endif

#include "test-atomic-ref.hpp"

// Basic MinMax

template <typename T>
class AtomicRefBasicMinMaxUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefBasicMinMaxUnitTest );

TYPED_TEST_P( AtomicRefBasicMinMaxUnitTest, BasicMinMaxs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T theval = (T)91;
  T * memaddr = &theval;
  T result;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test min
  result = test1.fetch_min( (T)87 );
  ASSERT_EQ( result, (T)91 );
  ASSERT_EQ( test1, (T)87 );

  result = test1.min( (T)83 );
  ASSERT_EQ( result, (T)83 );
  ASSERT_EQ( test1, (T)83 );

  // test max
  result = test1.fetch_max( (T)87 );
  ASSERT_EQ( result, (T)83 );
  ASSERT_EQ( test1, (T)87 );

  result = test1.max( (T)91 );
  ASSERT_EQ( result, (T)91 );
  ASSERT_EQ( test1, (T)91 );
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefBasicMinMaxUnitTest,
                             BasicMinMaxs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicMinMaxUnitTest,
                                AtomicRefBasicMinMaxUnitTest,
                                basic_types
                              );

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
// CUDA Accessors

template <typename T>
class AtomicRefCUDAMinMaxUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( AtomicRefCUDAMinMaxUnitTest );

GPU_TYPED_TEST_P( AtomicRefCUDAMinMaxUnitTest, CUDAMinMaxs )
{
  using T = typename std::tuple_element<0, TypeParam>::type;
  using AtomicPolicy = typename std::tuple_element<1, TypeParam>::type;

  T * result = nullptr;
  T * memaddr = nullptr;
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaMallocManaged, &result, sizeof(T));
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaMallocManaged, &memaddr, sizeof(T));
  memaddr[0] = (T)91;
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test min
  forone<test_cuda>( [=] __device__ () {result[0] = test1.fetch_min( (T)87 );} );
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)91 );
  ASSERT_EQ( test1, (T)87 );

  forone<test_cuda>( [=] __device__ () {result[0] = test1.min( (T)83 );} );
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)83 );
  ASSERT_EQ( test1, (T)83 );

  // test max
  forone<test_cuda>( [=] __device__ () {result[0] = test1.fetch_max( (T)87 );} );
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)83 );
  ASSERT_EQ( test1, (T)87 );

  forone<test_cuda>( [=] __device__ () {result[0] = test1.max( (T)91 );} );
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);
  ASSERT_EQ( result[0], (T)91 );
  ASSERT_EQ( test1, (T)91 );

  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaDeviceSynchronize);
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaFree, result);
  RAJA_INTERNAL_CUDA_CHECK_API_CALL(cudaFree, memaddr);
}

REGISTER_TYPED_TEST_SUITE_P( AtomicRefCUDAMinMaxUnitTest,
                             CUDAMinMaxs
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( CUDAMinMaxUnitTest,
                                AtomicRefCUDAMinMaxUnitTest,
                                CUDA_types
                              );
#endif

