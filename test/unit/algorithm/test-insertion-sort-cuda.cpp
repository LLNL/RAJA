//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA util insertion_sort for cuda gpus
///

#include "test-sort.hpp"

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Sort, insertion_Sort_cuda)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  testSorter(InsertionSortGPU<forone_cuda>{}, MaxN);
  testSorter(InsertionSortPairsGPU<forone_cuda>{}, MaxN);
}

#endif

