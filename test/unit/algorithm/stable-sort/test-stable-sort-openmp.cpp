//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA stable_sort with openmp policies
///

#include "../test-sort.hpp"

#if defined(RAJA_ENABLE_OPENMP)

TEST(Sort, StableSort_openmp)
{
  testSorter(PolicyStableSort<RAJA::omp_parallel_for_exec>{"omp"});
  testSorter(PolicyStableSortPairs<RAJA::omp_parallel_for_exec>{"omp"});
}

#endif

