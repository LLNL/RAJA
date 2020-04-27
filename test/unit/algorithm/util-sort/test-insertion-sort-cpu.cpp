//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util insertion_sort on the cpu
///

#include "../test-sort.hpp"

TEST(Sort, insertion_Sort_cpu)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(InsertionSort{}, MaxN);
  testSorter(InsertionSortPairs{}, MaxN);
}

