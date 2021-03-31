//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Graph Constructors
///

#include "RAJA_test-base.hpp"

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#include "RAJA_test-graph-execpol.hpp"
#include "RAJA_test-forall-execpol.hpp"


// Basic Constructors

template <typename T>
class GraphBasicConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest );

TYPED_TEST_P( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using forallPolicy = typename camp::at<TypeParam, camp::num<1>>::type;

  // default constructor
  RAJA::expt::graph::DAG<GraphPolicy> test1;

  // test ()
  ASSERT_TRUE( test1.empty() );

  RAJA::TypedRangeSegment<int> seg(0, 10);

  test1.template emplace_forall<forallPolicy>(seg, [=](int i) {

  });

  ASSERT_FALSE( test1.empty() );



}

//
// Cartesian product of types used in parameterized tests
//
using SequentialResourceTypes =
  Test< camp::cartesian_product<SequentialGraphExecPols,
                                SequentialForallExecPols>>::Types;


REGISTER_TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest,
                             BasicConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicConstructorUnitTest,
                                GraphBasicConstructorUnitTest,
                                SequentialResourceTypes
                              );
