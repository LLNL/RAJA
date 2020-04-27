//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util insertion_sort for hip gpus
///

#include "../test-sort.hpp"

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Sort, insertion_Sort_hip)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  testSorter(InsertionSortGPU<forone_hip>{}, MaxN);
  testSorter(InsertionSortPairsGPU<forone_hip>{}, MaxN);
}

#endif

