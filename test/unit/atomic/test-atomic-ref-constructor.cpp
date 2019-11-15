//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for atomic ref constructors (and use of getPointer for verification)
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

template <typename T>
void testAtomicDefaultPolConstructors()
{
  T * memaddr = nullptr;

  // explicit constructor with memory address
  RAJA::AtomicRef<T> test1( memaddr );

  ASSERT_EQ( test1.getPointer(), nullptr );

  // ref constructor
  RAJA::AtomicRef<T> const & reft1 = test1;
  RAJA::AtomicRef<T> reftest1( reft1 );

  ASSERT_EQ( reftest1.getPointer(), nullptr );
}

TEST( AtomicRefUnitTest, DefaultPolConstructorsTest )
{
  testAtomicDefaultPolConstructors<int>();
  testAtomicDefaultPolConstructors<float>();
  testAtomicDefaultPolConstructors<double>();
}

template <typename T, typename AtomicPolicy>
void testAtomicBasicConstructors()
{
  T * memaddr = nullptr;

  // explicit constructor with memory address
  RAJA::AtomicRef<T, AtomicPolicy> test1( memaddr );

  ASSERT_EQ( test1.getPointer(), nullptr );

  // ref constructor
  RAJA::AtomicRef<T, AtomicPolicy> const & reft1 = test1;
  RAJA::AtomicRef<T, AtomicPolicy> reftest1( reft1 );

  ASSERT_EQ( reftest1.getPointer(), nullptr );
}

TEST( AtomicRefUnitTest, BasicConstructorsTest )
{
  testAtomicBasicConstructors<int, RAJA::builtin_atomic>();
  testAtomicBasicConstructors<int, RAJA::seq_atomic>();

  testAtomicBasicConstructors<float, RAJA::builtin_atomic>();
  testAtomicBasicConstructors<float, RAJA::seq_atomic>();

  testAtomicBasicConstructors<double, RAJA::builtin_atomic>();
  testAtomicBasicConstructors<double, RAJA::seq_atomic>();

  #if defined(RAJA_ENABLE_OPENMP)
  testAtomicBasicConstructors<int, RAJA::omp_atomic>();

  testAtomicBasicConstructors<float, RAJA::omp_atomic>();

  testAtomicBasicConstructors<double, RAJA::omp_atomic>();
  #endif

  #if defined(RAJA_ENABLE_CUDA)
  testAtomicBasicConstructors<int, RAJA::auto_atomic>();
  testAtomicBasicConstructors<int, RAJA::cuda_atomic>();

  testAtomicBasicConstructors<float, RAJA::auto_atomic>();
  testAtomicBasicConstructors<float, RAJA::cuda_atomic>();

  testAtomicBasicConstructors<double, RAJA::auto_atomic>();
  testAtomicBasicConstructors<double, RAJA::cuda_atomic>();
  #endif
}
