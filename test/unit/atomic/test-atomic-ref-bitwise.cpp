//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic bit methods
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

template <typename T, typename AtomicPolicy>
void testAtomicBitwise()
{
  T theval = (T)1;
  T * memaddr = &theval;
  T result;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  // test and/or
  result = test1.fetch_and( (T)0 );
  ASSERT_EQ( result, (T)1 );
  ASSERT_EQ( test1, (T)0 );

  result = test1.fetch_or( (T)1 );
  ASSERT_EQ( result, (T)0 );
  ASSERT_EQ( test1, (T)1 );

  result = (test1 &= (T)0);
  ASSERT_EQ( test1, (T)0 );
  ASSERT_EQ( result, (T)0 );

  result = (test1 |= (T)1);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result, (T)1 );

  // test xor
  result = test1.fetch_xor( (T)1 );
  ASSERT_EQ( result, (T)1 );
  ASSERT_EQ( test1, (T)0 );

  result = (test1 ^= (T)1);
  ASSERT_EQ( test1, (T)1 );
  ASSERT_EQ( result, (T)1 );
}

TEST( AtomicRefUnitTest, BitwiseTest )
{
  // NOTE: Need to revisit auto_atomic and cuda policies which use pointers
  //testAtomicBitwise<int, RAJA::auto_atomic>();
  //testAtomicBitwise<int, RAJA::cuda_atomic>();
  testAtomicBitwise<unsigned int, RAJA::omp_atomic>();
  testAtomicBitwise<unsigned int, RAJA::builtin_atomic>();
  testAtomicBitwise<unsigned int, RAJA::seq_atomic>();

  // Hopefully nobody will try these . . .
  //testAtomicBitwise<float, RAJA::auto_atomic>();
  //testAtomicBitwise<float, RAJA::cuda_atomic>();
  //testAtomicBitwise<float, RAJA::omp_atomic>();
  //testAtomicBitwise<float, RAJA::builtin_atomic>();
  //testAtomicBitwise<float, RAJA::seq_atomic>();

  //testAtomicBitwise<double, RAJA::auto_atomic>();
  //testAtomicBitwise<double, RAJA::cuda_atomic>();
  //testAtomicBitwise<double, RAJA::omp_atomic>();
  //testAtomicBitwise<double, RAJA::builtin_atomic>();
  //testAtomicBitwise<double, RAJA::seq_atomic>();
}

