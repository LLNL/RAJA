//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic add, subtract, inc, and dec methods
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit_forone.hpp"
#endif

template <typename T, typename AtomicPolicy>
void testAtomicAddSub()
{
  T theval = (T)0;
  T * memaddr = &theval;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test inc ops
  T val2 = ++test1;
  T val3 = test1++;
  ASSERT_EQ( test1, (T)2 );
  ASSERT_EQ( val2, (T)1 );
  ASSERT_EQ( val3, (T)1 );

  // test dec ops
  T val4 = --test1;
  T val5 = test1--;
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( val4, (T)1 );
  ASSERT_EQ( val5, (T)1 );

  // test add/sub ops
  T val6 = (test1 += (T)23);
  ASSERT_EQ( test1, (T)23 );
  ASSERT_EQ( val6, (T)23 );
  T val7 = (test1 -= (T)22);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( val7, (T)1 );

  // test add/sub methods
  T val8 = test1.fetch_add( (T)23 );
  ASSERT_EQ( test1, (T)24 );
  ASSERT_EQ( val8, (T)1 );
  T val9 = test1.fetch_sub( (T)23 );
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( val9, (T)24 );
}

// Pure CUDA test.
#if defined(RAJA_ENABLE_CUDA)
template <typename T, typename AtomicPolicy>
void testAtomicAddSubCUDA()
{
  T * memaddr = nullptr;
  T * result1 = nullptr;
  T * result2 = nullptr;
  cudaErrchk(cudaMallocManaged((void **)&memaddr, sizeof(T)));
  cudaErrchk(cudaMallocManaged((void **)&result1, sizeof(T)));
  cudaErrchk(cudaMallocManaged((void **)&result2, sizeof(T)));
  memaddr[0] = (T)0;
  cudaErrchk(cudaDeviceSynchronize());

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test inc ops
  forone<<<1,1>>>( [=] __device__ () {result1[0] = ++test1;} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( result1[0], (T)1 );
  forone<<<1,1>>>( [=] __device__ () {result2[0] = test1++;} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)2 );
  ASSERT_EQ( result2[0], (T)1 );

  // test dec ops
  forone<<<1,1>>>( [=] __device__ () {result1[0] = --test1;} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( result1[0], (T)1 );
  forone<<<1,1>>>( [=] __device__ () {result2[0] = test1--;} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( result2[0], (T)1 );

  // test add/sub ops
  forone<<<1,1>>>( [=] __device__ () {result1[0] = (test1 += (T)23);} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)23 );
  ASSERT_EQ( result1[0], (T)23 );
  forone<<<1,1>>>( [=] __device__ () {result2[0] = (test1 -= (T)22);} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result2[0], (T)1 );

  // test add/sub methods
  forone<<<1,1>>>( [=] __device__ () {result1[0] = test1.fetch_add( (T)23 );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)24 );
  ASSERT_EQ( result1[0], (T)1 );
  forone<<<1,1>>>( [=] __device__ () {result2[0] = test1.fetch_sub( (T)23 );} );
  cudaErrchk(cudaDeviceSynchronize());
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result2[0], (T)24 );

  cudaErrchk(cudaDeviceSynchronize());
  cudaErrchk(cudaFree(memaddr));
  cudaErrchk(cudaFree(result1));
  cudaErrchk(cudaFree(result2));
}
#endif

TEST( AtomicRefUnitTest, AddSubTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  testAtomicAddSub<int, RAJA::builtin_atomic>();
  testAtomicAddSub<int, RAJA::seq_atomic>();

  testAtomicAddSub<unsigned int, RAJA::builtin_atomic>();
  testAtomicAddSub<unsigned int, RAJA::seq_atomic>();

  testAtomicAddSub<unsigned long long int, RAJA::builtin_atomic>();
  testAtomicAddSub<unsigned long long int, RAJA::seq_atomic>();

  testAtomicAddSub<float, RAJA::builtin_atomic>();
  testAtomicAddSub<float, RAJA::seq_atomic>();

  testAtomicAddSub<double, RAJA::builtin_atomic>();
  testAtomicAddSub<double, RAJA::seq_atomic>();

  #if defined(RAJA_ENABLE_OPENMP)
  testAtomicAddSub<int, RAJA::omp_atomic>();
  testAtomicAddSub<unsigned int, RAJA::omp_atomic>();
  testAtomicAddSub<unsigned long long int, RAJA::omp_atomic>();
  testAtomicAddSub<float, RAJA::omp_atomic>();
  testAtomicAddSub<double, RAJA::omp_atomic>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  testAtomicAddSub<int, RAJA::auto_atomic>();
  testAtomicAddSub<unsigned int, RAJA::auto_atomic>();
  testAtomicAddSub<unsigned long long int, RAJA::auto_atomic>();
  testAtomicAddSub<float, RAJA::auto_atomic>();
  testAtomicAddSub<double, RAJA::auto_atomic>();

  testAtomicAddSub<int, RAJA::cuda_atomic>();
  testAtomicAddSub<unsigned int, RAJA::cuda_atomic>();
  testAtomicAddSub<unsigned long long int, RAJA::cuda_atomic>();
  testAtomicAddSub<float, RAJA::cuda_atomic>();
  testAtomicAddSub<double, RAJA::cuda_atomic>();

  testAtomicAddSubCUDA<int, RAJA::auto_atomic>();
  testAtomicAddSubCUDA<unsigned int, RAJA::auto_atomic>();
  testAtomicAddSubCUDA<unsigned long long int, RAJA::auto_atomic>();
  testAtomicAddSubCUDA<float, RAJA::auto_atomic>();
  testAtomicAddSubCUDA<double, RAJA::auto_atomic>();

  testAtomicAddSubCUDA<int, RAJA::cuda_atomic>();
  testAtomicAddSubCUDA<unsigned int, RAJA::cuda_atomic>();
  testAtomicAddSubCUDA<unsigned long long int, RAJA::cuda_atomic>();
  testAtomicAddSubCUDA<float, RAJA::cuda_atomic>();
  testAtomicAddSubCUDA<double, RAJA::cuda_atomic>();
  #endif
}

