//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for span
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include <type_traits>


template <typename ValueType, typename IndexType>
void testSpanConstructTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  {
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);

    ASSERT_EQ(ptr, span.data());
    ASSERT_EQ(len, span.size());
  }

  {
    const RAJA::Span<ValueType*, IndexType> span(ptr, ptr+len);

    ASSERT_EQ(ptr, span.data());
    ASSERT_EQ(len, span.size());
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanAssignTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  {
    RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> span2(ptr, len);
    span = span2;

    ASSERT_EQ(ptr, span.data());
    ASSERT_EQ(len, span.size());
  }

  {
    ValueType* ptr2 = ptr + 1;
    IndexType len2 = 1;
    RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> span2(ptr2, len2);
    span = span2;

    ASSERT_EQ(ptr2, span.data());
    ASSERT_EQ(len2, span.size());
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanIteratorTypes()
{
  using span_type = RAJA::Span<ValueType*, IndexType>;
  using iterator = typename span_type::iterator;
  using const_iterator = typename span_type::const_iterator;
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  // XL cannot handle initialization list with new
  // e.g. new ValueType[len]{0,1,2,3} produces error
  for ( int ii = 0; ii < static_cast<int>(len); ++ii )
  {
    ptr[ii] = ii;
  }

  {
    const span_type span(ptr, len);

    iterator begin = span.begin();
    iterator end = span.end();
    ASSERT_EQ(ptr, begin);
    ASSERT_EQ(ptr+len, end);

    ValueType* ptr_chk = ptr;

    for (iterator iter = begin; iter != end; ++iter) {
      ASSERT_EQ(*ptr_chk, *iter);
      ptr_chk++ ;
    }

    const_iterator cbegin = span.cbegin();
    const_iterator cend = span.cend();
    ASSERT_EQ(ptr, cbegin);
    ASSERT_EQ(ptr+len, cend);

    ptr_chk = ptr;

    for (iterator citer = cbegin; citer != cend; ++citer) {
      ASSERT_EQ(*ptr_chk, *citer);
      ptr_chk++ ;
    }
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanElementAccessTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  // XL cannot handle initialization list with new
  // e.g. new ValueType[len]{0,1,2,3} produces error
  for ( int ii = 0; ii < static_cast<int>(len); ++ii )
  {
    ptr[ii] = ii;
  }

  {
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);

    ASSERT_EQ(ptr, span.data());
    ASSERT_EQ(*ptr, span.front());
    ASSERT_EQ(*(ptr+len-1), span.back());

    for (IndexType i = 0; i < len; ++i) {
      ASSERT_EQ(ptr[i], span[i]);
    }
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanObserveTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  // XL cannot handle initialization list with new
  // e.g. new ValueType[len]{0,1,2,3} produces error
  for ( int ii = 0; ii < static_cast<int>(len); ++ii )
  {
    ptr[ii] = ii;
  }

  {
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);

    ASSERT_EQ(len, span.size());
    ASSERT_FALSE(span.empty());
  }

  {
    const RAJA::Span<ValueType*, IndexType> span(ptr, len-len);

    ASSERT_EQ(0, span.size());
    ASSERT_TRUE(span.empty());
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanSubViewTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  // XL cannot handle initialization list with new
  // e.g. new ValueType[len]{0,1,2,3} produces error
  for ( int ii = 0; ii < static_cast<int>(len); ++ii )
  {
    ptr[ii] = ii;
  }

  {
    IndexType count = 3;
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> subspan = span.first(count);

    ASSERT_EQ(count, subspan.size());
    ASSERT_EQ(ptr, subspan.data());
  }

  {
    IndexType count = 3;
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> subspan = span.last(count);

    ASSERT_EQ(count, subspan.size());
    ASSERT_EQ(ptr+len-count, subspan.data());
  }

  {
    IndexType begin = 1;
    IndexType count = 2;
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> subspan = span.subspan(begin, count);

    ASSERT_EQ(count, subspan.size());
    ASSERT_EQ(ptr+begin, subspan.data());
  }

  {
    IndexType begin = 1;
    IndexType count = 2;
    const RAJA::Span<ValueType*, IndexType> span(ptr, len);
    const RAJA::Span<ValueType*, IndexType> subspan = span.slice(begin, count);

    ASSERT_EQ(count, subspan.size());
    ASSERT_EQ(ptr+begin, subspan.data());
  }

  delete[] ptr;
}

template <typename ValueType, typename IndexType>
void testSpanMakeSpanTypes()
{
  IndexType len = 4;
  ValueType* ptr = new ValueType[len];

  {
    const RAJA::Span<ValueType*, IndexType> span = RAJA::make_span(ptr, len);

    ASSERT_EQ(ptr, span.data());
    ASSERT_EQ(len, span.size());
  }

  delete[] ptr;
}
