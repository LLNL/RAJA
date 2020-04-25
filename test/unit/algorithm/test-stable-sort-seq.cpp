//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA stable_sort with sequential policies
///

#include "test-sort.hpp"

TEST(Sort, StableSort_loop)
{
  testSorter(PolicyStableSort<RAJA::loop_exec>{"loop"});
  testSorter(PolicyStableSortPairs<RAJA::loop_exec>{"loop"});
}

TEST(Sort, StableSort_seq)
{
  testSorter(PolicyStableSort<RAJA::seq_exec>{"seq"});
  testSorter(PolicyStableSortPairs<RAJA::seq_exec>{"seq"});
}

