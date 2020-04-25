//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA stable_sort with tbb policies
///

#include "test-sort.hpp"

#if defined(RAJA_ENABLE_TBB)

TEST(Sort, StableSort_tbb)
{
  testSorter(PolicyStableSort<RAJA::tbb_for_exec>{"tbb"});
  testSorter(PolicyStableSortPairs<RAJA::tbb_for_exec>{"tbb"});
}

#endif

