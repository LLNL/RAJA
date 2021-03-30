//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Graph Constructors
///

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

// Basic Constructors

template <typename T>
class GraphBasicConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest );

TYPED_TEST_P( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = typename std::tuple_element<0, TypeParam>::type;

  // explicit default constructor
  RAJA::expt::graph::DAG<GraphPolicy> test1( );

  // test ()
  ASSERT_TRUE( test1.empty() );
}


using basic_types =
    ::testing::Types<
                      std::tuple<RAJA::seq_exec>
                     >;

REGISTER_TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest,
                             BasicConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicConstructorUnitTest,
                                GraphBasicConstructorUnitTest,
                                basic_types
                              );
