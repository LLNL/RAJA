//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic min and max methods
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

template <typename T, typename AtomicPolicy>
void testAtomicMinMax()
{
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

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
template <typename T, typename AtomicPolicy>
void testAtomicMinMaxCUDA()
{
  T * memaddr = nullptr;
  cudaErrchk(cudaMallocManaged(&memaddr, sizeof(T)));
  memaddr[0] = (T)91;
  cudaErrchk(cudaDeviceSynchronize());
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

  cudaErrchk(cudaFree(memaddr));
  cudaErrchk(cudaDeviceSynchronize());
}
#endif

TEST( AtomicRefUnitTest, MinMaxTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  testAtomicMinMax<int, RAJA::builtin_atomic>();
  testAtomicMinMax<int, RAJA::seq_atomic>();

  testAtomicMinMax<unsigned int, RAJA::builtin_atomic>();
  testAtomicMinMax<unsigned int, RAJA::seq_atomic>();

  testAtomicMinMax<unsigned long long int, RAJA::builtin_atomic>();
  testAtomicMinMax<unsigned long long int, RAJA::seq_atomic>();

  testAtomicMinMax<float, RAJA::builtin_atomic>();
  testAtomicMinMax<float, RAJA::seq_atomic>();

  testAtomicMinMax<double, RAJA::builtin_atomic>();
  testAtomicMinMax<double, RAJA::seq_atomic>();

  #if defined(RAJA_ENABLE_OPENMP)
  testAtomicMinMax<int, RAJA::omp_atomic>();
  testAtomicMinMax<unsigned int, RAJA::omp_atomic>();
  testAtomicMinMax<unsigned long long int, RAJA::omp_atomic>();
  testAtomicMinMax<float, RAJA::omp_atomic>();
  testAtomicMinMax<double, RAJA::omp_atomic>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  testAtomicMinMax<int, RAJA::auto_atomic>();
  testAtomicMinMax<int, RAJA::cuda_atomic>();

  testAtomicMinMax<unsigned int, RAJA::auto_atomic>();
  testAtomicMinMax<unsigned int, RAJA::cuda_atomic>();

  testAtomicMinMax<unsigned long long int, RAJA::auto_atomic>();
  testAtomicMinMax<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicMinMax<float, RAJA::auto_atomic>();
  testAtomicMinMax<float, RAJA::cuda_atomic>();

  testAtomicMinMax<double, RAJA::auto_atomic>();
  testAtomicMinMax<double, RAJA::cuda_atomic>();

  testAtomicMinMaxCUDA<int, RAJA::auto_atomic>();
  testAtomicMinMaxCUDA<int, RAJA::cuda_atomic>();

  testAtomicMinMaxCUDA<unsigned int, RAJA::auto_atomic>();
  testAtomicMinMaxCUDA<unsigned int, RAJA::cuda_atomic>();

  testAtomicMinMaxCUDA<unsigned long long int, RAJA::auto_atomic>();
  testAtomicMinMaxCUDA<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicMinMaxCUDA<float, RAJA::auto_atomic>();
  testAtomicMinMaxCUDA<float, RAJA::cuda_atomic>();

  testAtomicMinMaxCUDA<double, RAJA::auto_atomic>();
  testAtomicMinMaxCUDA<double, RAJA::cuda_atomic>();
  #endif
}

