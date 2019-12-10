//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic accessor methods
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit_forone.hpp"
#endif

template <typename T, typename AtomicPolicy>
void testAtomicAccessors()
{
  // should also work with CUDA
  T theval = (T)0;
  T * memaddr = &theval;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test store method with op()
  test1.store( (T)19 );
  ASSERT_EQ( test1, 19 );

  // test assignment operator
  test1 = (T)23;
  ASSERT_EQ( test1, 23 );

  // test load method
  test1 = (T)29;
  ASSERT_EQ( test1.load(), 29 );
}

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
template <typename T, typename AtomicPolicy>
void testAtomicAccessorsCUDA()
{
  T * memaddr = nullptr;
  T * result = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&memaddr, sizeof(T)));
  cudaErrchk(cudaMallocManaged((void **)&result, sizeof(T)));
  cudaErrchk(cudaDeviceSynchronize());

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test store method with op()
  forone<<<1,1>>>( [=] __device__ () {test1.store( (T)19 );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)19 );

  // test assignment operator
  forone<<<1,1>>>( [=] __device__ () {test1 = (T)23;} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)23 );

  // test load method
  forone<<<1,1>>>( [=] __device__ () {test1 = (T)29; result[0] = test1.load();} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( result[0], (T)29 );

  cudaErrchk(cudaDeviceSynchronize());

  cudaErrchk(cudaFree(memaddr));
  cudaErrchk(cudaFree(result));
}
#endif

TEST( AtomicRefUnitTest, AccessorsTest )
{
  testAtomicAccessors<int, RAJA::builtin_atomic>();
  testAtomicAccessors<int, RAJA::seq_atomic>();

  testAtomicAccessors<unsigned int, RAJA::builtin_atomic>();
  testAtomicAccessors<unsigned int, RAJA::seq_atomic>();

  testAtomicAccessors<unsigned long long int, RAJA::builtin_atomic>();
  testAtomicAccessors<unsigned long long int, RAJA::seq_atomic>();

  testAtomicAccessors<float, RAJA::builtin_atomic>();
  testAtomicAccessors<float, RAJA::seq_atomic>();

  testAtomicAccessors<double, RAJA::builtin_atomic>();
  testAtomicAccessors<double, RAJA::seq_atomic>();

  #if defined(RAJA_ENABLE_OPENMP)
  testAtomicAccessors<int, RAJA::omp_atomic>();

  testAtomicAccessors<unsigned int, RAJA::omp_atomic>();

  testAtomicAccessors<unsigned long long int, RAJA::omp_atomic>();

  testAtomicAccessors<float, RAJA::omp_atomic>();

  testAtomicAccessors<double, RAJA::omp_atomic>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  testAtomicAccessors<int, RAJA::auto_atomic>();
  testAtomicAccessors<int, RAJA::cuda_atomic>();

  testAtomicAccessors<unsigned int, RAJA::auto_atomic>();
  testAtomicAccessors<unsigned int, RAJA::cuda_atomic>();

  testAtomicAccessors<unsigned long long int, RAJA::auto_atomic>();
  testAtomicAccessors<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicAccessors<float, RAJA::auto_atomic>();
  testAtomicAccessors<float, RAJA::cuda_atomic>();

  testAtomicAccessors<double, RAJA::auto_atomic>();
  testAtomicAccessors<double, RAJA::cuda_atomic>();

  testAtomicAccessorsCUDA<int, RAJA::auto_atomic>();
  testAtomicAccessorsCUDA<int, RAJA::cuda_atomic>();

  testAtomicAccessorsCUDA<unsigned int, RAJA::auto_atomic>();
  testAtomicAccessorsCUDA<unsigned int, RAJA::cuda_atomic>();

  testAtomicAccessorsCUDA<unsigned long long int, RAJA::auto_atomic>();
  testAtomicAccessorsCUDA<unsigned long long int, RAJA::cuda_atomic>();

  testAtomicAccessorsCUDA<float, RAJA::auto_atomic>();
  testAtomicAccessorsCUDA<float, RAJA::cuda_atomic>();

  testAtomicAccessorsCUDA<double, RAJA::auto_atomic>();
  testAtomicAccessorsCUDA<double, RAJA::cuda_atomic>();
  #endif
}

