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
  test1 += (T)23;
  ASSERT_EQ( test1, (T)23 );
  test1 -= (T)22;
  ASSERT_EQ( test1, (T)1 );

  // test add/sub methods
  test1.fetch_add( (T)23 );
  ASSERT_EQ( test1, (T)24 );
  test1.fetch_sub( (T)23 );
  ASSERT_EQ( test1, (T)1 );
}

TEST( AtomicRefUnitTest, AddSubTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  //testAtomicAddSub<int, RAJA::auto_atomic>();
  //testAtomicAddSub<int, RAJA::cuda_atomic>();
  testAtomicAddSub<int, RAJA::omp_atomic>();
  testAtomicAddSub<int, RAJA::builtin_atomic>();
  testAtomicAddSub<int, RAJA::seq_atomic>();

  //testAtomicAddSub<float, RAJA::auto_atomic>();
  //testAtomicAddSub<float, RAJA::cuda_atomic>();
  testAtomicAddSub<float, RAJA::omp_atomic>();
  testAtomicAddSub<float, RAJA::builtin_atomic>();
  testAtomicAddSub<float, RAJA::seq_atomic>();

  //testAtomicAddSub<double, RAJA::auto_atomic>();
  //testAtomicAddSub<double, RAJA::cuda_atomic>();
  testAtomicAddSub<double, RAJA::omp_atomic>();
  testAtomicAddSub<double, RAJA::builtin_atomic>();
  testAtomicAddSub<double, RAJA::seq_atomic>();
}

