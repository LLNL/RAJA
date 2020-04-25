//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with sequential policies
///

#include "test-sort.hpp"

TEST(Sort, Sort_loop)
{
  testSorter(PolicySort<RAJA::loop_exec>{"loop"});
  testSorter(PolicySortPairs<RAJA::loop_exec>{"loop"});
}

TEST(Sort, Sort_seq)
{
  testSorter(PolicySort<RAJA::seq_exec>{"seq"});
  testSorter(PolicySortPairs<RAJA::seq_exec>{"seq"});
}

