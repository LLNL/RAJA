//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Span
///

#include "test-span.hpp"

#define RAJA_SPAN_RUN_TEST(test) \
  test<int, int>(); \
  test<int, size_t>(); \
  test<double, int>(); \
  test<double, size_t>(); \

TEST(Span, basic_construct_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanConstructTypes)
}

TEST(Span, basic_assign_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanAssignTypes)
}

TEST(Span, basic_iterator_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanIteratorTypes)
}

TEST(Span, basic_element_access_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanElementAccessTypes)
}

TEST(Span, basic_observe_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanObserveTypes)
}

TEST(Span, basic_subview_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanSubViewTypes)
}

TEST(Span, basic_make_span_Span)
{
  RAJA_SPAN_RUN_TEST(testSpanMakeSpanTypes)
}
