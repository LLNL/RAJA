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

  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  // default constructor
  graph_type test1;

  // test ()
  ASSERT_TRUE( test1.empty() );

  RAJA::expt::graph::EmptyNode* node1 = make_EmptyNode(test1);

  ASSERT_TRUE( node1 != nullptr );

  ASSERT_FALSE( test1.empty() );

  auto r = Res::get_default();
  test1.exec(r);
}

//
// Cartesian product of types used in parameterized tests
//
using ResourceTypes =
  Test< camp::cartesian_product<SequentialGraphExecPols>>::Types;


REGISTER_TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest,
                             BasicConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicConstructorUnitTest,
                                GraphBasicConstructorUnitTest,
                                ResourceTypes
                              );
