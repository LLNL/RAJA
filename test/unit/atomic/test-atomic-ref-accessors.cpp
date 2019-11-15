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

template <typename T, typename AtomicPolicy>
void testAtomicAccessors()
{
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

TEST( AtomicRefUnitTest, AccessorsTest )
{
  testAtomicAccessors<int, RAJA::auto_atomic>();
  testAtomicAccessors<int, RAJA::cuda_atomic>();
  testAtomicAccessors<int, RAJA::omp_atomic>();
  testAtomicAccessors<int, RAJA::builtin_atomic>();
  testAtomicAccessors<int, RAJA::seq_atomic>();

  testAtomicAccessors<float, RAJA::auto_atomic>();
  testAtomicAccessors<float, RAJA::cuda_atomic>();
  testAtomicAccessors<float, RAJA::omp_atomic>();
  testAtomicAccessors<float, RAJA::builtin_atomic>();
  testAtomicAccessors<float, RAJA::seq_atomic>();

  testAtomicAccessors<double, RAJA::auto_atomic>();
  testAtomicAccessors<double, RAJA::cuda_atomic>();
  testAtomicAccessors<double, RAJA::omp_atomic>();
  testAtomicAccessors<double, RAJA::builtin_atomic>();
  testAtomicAccessors<double, RAJA::seq_atomic>();
}

