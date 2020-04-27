//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA stable_sort with hip policies
///

#include "../test-sort.hpp"

#if defined(RAJA_ENABLE_HIP)

TEST(Sort, StableSort_hip)
{
  testSorter(PolicyStableSort<RAJA::hip_exec<128>>{"hip"});
  testSorter(PolicyStableSortPairs<RAJA::hip_exec<128>>{"hip"});
}

#endif

