//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util shell_sort for cuda gpus
///

#include "test-sort.hpp"

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Sort, shell_Sort_cuda)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(ShellSortGPU<forone_cuda>{}, MaxN);
  testSorter(ShellSortPairsGPU<forone_cuda>{}, MaxN);
}

#endif

