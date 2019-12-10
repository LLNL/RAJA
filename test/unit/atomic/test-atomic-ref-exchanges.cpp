//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic exchange and swap methods
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit_forone.hpp"
#endif

template <typename T, typename AtomicPolicy>
void testAtomicExchanges()
{
  T swapper = (T)91;
  T theval = (T)0;
  T * memaddr = &theval;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test exchange method
  swapper = test1.exchange( swapper );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper, (T)0 );

  // test CAS method
  swapper = test1.CAS( (T)91, swapper );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper, (T)91 );


  bool result = true;
  T testval = (T)19;
  T & valref = testval;

  // test strong exchange method
  result = test1.compare_exchange_strong( valref, testval );
  ASSERT_EQ( result, false );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper, (T)91 );
  ASSERT_EQ( testval, (T)0 );

  // test weak exchange method (same as strong exchange)
  result = test1.compare_exchange_weak( valref, swapper );
  ASSERT_EQ( result, true );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper, (T)91 );
  ASSERT_EQ( testval, (T)0 );
}

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
template <typename T, typename AtomicPolicy>
void testAtomicExchangesCUDA()
{
  T * swapper = nullptr;
  T * memaddr = nullptr;
  T * testval = nullptr;
  bool * result = nullptr;
  cudaErrchk(cudaMallocManaged(&swapper, sizeof(T)));
  cudaErrchk(cudaMallocManaged(&memaddr, sizeof(T)));
  cudaErrchk(cudaMallocManaged(&testval, sizeof(T)));
  cudaErrchk(cudaMallocManaged(&result, sizeof(bool)));
  swapper[0] = (T)91;
  memaddr[0] = (T)0;
  testval[0] = (T)19;
  result[0] = true;
  cudaErrchk(cudaDeviceSynchronize());

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test exchange method
  forone<<<1,1>>>( [=] __device__ () {swapper[0] = test1.exchange( swapper[0] );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper[0], (T)0 );

  // test CAS method
  forone<<<1,1>>>( [=] __device__ () {swapper[0] = test1.CAS( (T)91, swapper[0] );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper[0], (T)91 );

  // test strong exchange method
  forone<<<1,1>>>( [=] __device__ () {result[0] = test1.compare_exchange_strong( testval[0], testval[0] );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( result[0], false );
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( swapper[0], (T)91 );
  ASSERT_EQ( testval[0], (T)0 );

  // test weak exchange method (same as strong exchange)
  forone<<<1,1>>>( [=] __device__ () {result[0] = test1.compare_exchange_weak( testval[0], swapper[0] );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( result[0], true );
  ASSERT_EQ( test1, (T)91 );
  ASSERT_EQ( swapper[0], (T)91 );
  ASSERT_EQ( testval[0], (T)0 );

  cudaErrchk(cudaDeviceSynchronize());
  cudaErrchk(cudaFree(swapper));
  cudaErrchk(cudaFree(memaddr));
  cudaErrchk(cudaFree(testval));
  cudaErrchk(cudaFree(result));
}
#endif

TEST( AtomicRefUnitTest, ExchangesTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  testAtomicExchanges<int, RAJA::builtin_atomic>();
  testAtomicExchanges<int, RAJA::seq_atomic>();

  testAtomicExchanges<unsigned int, RAJA::builtin_atomic>();
  testAtomicExchanges<unsigned int, RAJA::seq_atomic>();

  testAtomicExchanges<unsigned long long int, RAJA::builtin_atomic>();
  testAtomicExchanges<unsigned long long int, RAJA::seq_atomic>();

  testAtomicExchanges<float, RAJA::builtin_atomic>();
  testAtomicExchanges<float, RAJA::seq_atomic>();

  testAtomicExchanges<double, RAJA::builtin_atomic>();
  testAtomicExchanges<double, RAJA::seq_atomic>();

  #if defined(RAJA_ENABLE_OPENMP)
  testAtomicExchanges<int, RAJA::omp_atomic>();
  testAtomicExchanges<unsigned int, RAJA::omp_atomic>();
  testAtomicExchanges<unsigned long long int, RAJA::omp_atomic>();
  testAtomicExchanges<float, RAJA::omp_atomic>();
  testAtomicExchanges<double, RAJA::omp_atomic>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  testAtomicExchanges<int, RAJA::auto_atomic>();
  testAtomicExchanges<int, RAJA::cuda_atomic>();

  testAtomicExchanges<unsigned int, RAJA::auto_atomic>();
  testAtomicExchanges<unsigned int, RAJA::cuda_atomic>();

  testAtomicExchanges<unsigned long long int, RAJA::auto_atomic>();
  testAtomicExchanges<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicExchanges<float, RAJA::auto_atomic>();
  testAtomicExchanges<float, RAJA::cuda_atomic>();

  testAtomicExchangesCUDA<int, RAJA::auto_atomic>();
  testAtomicExchangesCUDA<int, RAJA::cuda_atomic>();

  testAtomicExchangesCUDA<unsigned int, RAJA::auto_atomic>();
  testAtomicExchangesCUDA<unsigned int, RAJA::cuda_atomic>();

  testAtomicExchangesCUDA<unsigned long long int, RAJA::auto_atomic>();
  testAtomicExchangesCUDA<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicExchangesCUDA<float, RAJA::auto_atomic>();
  testAtomicExchangesCUDA<float, RAJA::cuda_atomic>();
  #endif
}

