//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Types and type lists for loop indexing used throughout RAJA tests.
//
// Note that in the type lists, a subset of types is used by default.
// For more comprehensive type testing define the macro RAJA_TEST_EXHAUSTIVE.
//

#ifndef __RAJA_test_atomic_types_HPP__
#define __RAJA_test_atomic_types_HPP__

#include "camp/list.hpp"

//
// Atomic data types
//
using AtomicDataTypeList =
  camp::list< RAJA::Index_type,
              int,
#if defined(RAJA_TEST_EXHAUSTIVE)
              unsigned,
              long long,
              unsigned long long,
              float,
#endif
              double >;

using AtomicSegmentList =
  camp::list< RAJA::TypedRangeSegment<RAJA::Index_type>,
              RAJA::TypedRangeStrideSegment<RAJA::Index_type>,
              RAJA::TypedListSegment<RAJA::Index_type> >;

#endif // __RAJA_test_atomic_types_HPP__
