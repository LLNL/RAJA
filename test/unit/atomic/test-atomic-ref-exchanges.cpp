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
  T swapper = (T)91;
  T * memaddr = nullptr;
  cudaErrchk(cudaMallocManaged(&memaddr, sizeof(T)));
  memaddr[0] = (T)0;
  cudaErrchk(cudaDeviceSynchronize());

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

  cudaErrchk(cudaDeviceSynchronize());
  cudaErrchk(cudaFree(memaddr));
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

