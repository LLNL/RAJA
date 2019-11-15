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

TEST( AtomicRefUnitTest, MinMaxTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  //testAtomicMinMax<int, RAJA::auto_atomic>();
  //testAtomicMinMax<int, RAJA::cuda_atomic>();
  //testAtomicMinMax<int, RAJA::omp_atomic>();
  testAtomicMinMax<int, RAJA::builtin_atomic>();
  testAtomicMinMax<int, RAJA::seq_atomic>();

  //testAtomicMinMax<float, RAJA::auto_atomic>();
  //testAtomicMinMax<float, RAJA::cuda_atomic>();
  //testAtomicMinMax<float, RAJA::omp_atomic>();
  testAtomicMinMax<float, RAJA::builtin_atomic>();
  testAtomicMinMax<float, RAJA::seq_atomic>();

  //testAtomicMinMax<double, RAJA::auto_atomic>();
  //testAtomicMinMax<double, RAJA::cuda_atomic>();
  //testAtomicMinMax<double, RAJA::omp_atomic>();
  testAtomicMinMax<double, RAJA::builtin_atomic>();
  testAtomicMinMax<double, RAJA::seq_atomic>();
}

