//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with openmp policies
///

#include "../test-sort.hpp"

#if defined(RAJA_ENABLE_OPENMP)

TEST(Sort, Sort_openmp)
{
  testSorter(PolicySort<RAJA::omp_parallel_for_exec>{"omp"});
  testSorter(PolicySortPairs<RAJA::omp_parallel_for_exec>{"omp"});
}

#endif

