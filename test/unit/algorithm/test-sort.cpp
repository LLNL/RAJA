//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for sort
///

#include "test-sort.hpp"

TEST(Sort, basic_algorithm_Sort)
{
  testSorter(InsertionSort{});
  testSorter(InsertionSortPairs{});
  testSorter(ShellSort{});
  testSorter(ShellSortPairs{});
  testSorter(HeapSort{});
  testSorter(HeapSortPairs{});
  testSorter(IntroSort{});
  testSorter(IntroSortPairs{});
  testSorter(MergeSort{});
  // testSorter(MergeSortPairs{});
}

TEST(Sort, basic_loop_Sort)
{
  testSorter(PolicySort<RAJA::loop_exec>{});
  testSorter(PolicyStableSort<RAJA::loop_exec>{});
  testSorter(PolicySortPairs<RAJA::loop_exec>{});
  // testSorter(PolicyStableSortPairs<RAJA::loop_exec>{});
}

TEST(Sort, basic_seq_Sort)
{
  testSorter(PolicySort<RAJA::seq_exec>{});
  testSorter(PolicyStableSort<RAJA::seq_exec>{});
  testSorter(PolicySortPairs<RAJA::seq_exec>{});
  // testSorter(PolicyStableSortPairs<RAJA::seq_exec>{});
}

#if defined(RAJA_ENABLE_OPENMP)

TEST(Sort, basic_OpenMP_Sort)
{
  testSorter(PolicySort<RAJA::omp_parallel_for_exec>{});
  testSorter(PolicyStableSort<RAJA::omp_parallel_for_exec>{});
  // testSorter(PolicySortPairs<RAJA::omp_parallel_for_exec>{});
  // testSorter(PolicyStableSortPairs<RAJA::omp_parallel_for_exec>{});
}

#endif

#if defined(RAJA_ENABLE_TBB)

TEST(Sort, basic_TBB_Sort)
{
  testSorter(PolicySort<RAJA::tbb_for_exec>{});
  testSorter(PolicyStableSort<RAJA::tbb_for_exec>{});
  // testSorter(PolicySortPairs<RAJA::tbb_for_exec>{});
  // testSorter(PolicyStableSortPairs<RAJA::tbb_for_exec>{});
}
#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Sort, basic_CUDA_Sort)
{
  testSorter(PolicySort<RAJA::cuda_exec<256>>{});
  testSorter(PolicyStableSort<RAJA::cuda_exec<256>>{});
  testSorter(PolicySortPairs<RAJA::cuda_exec<256>>{});
  testSorter(PolicyStableSortPairs<RAJA::cuda_exec<256>>{});
}

#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Sort, basic_HIP_Sort)
{
  testSorter(PolicySort<RAJA::hip_exec<256>>{});
  testSorter(PolicyStableSort<RAJA::hip_exec<256>>{});
  testSorter(PolicySortPairs<RAJA::hip_exec<256>>{});
  testSorter(PolicyStableSortPairs<RAJA::hip_exec<256>>{});
}

#endif

#if defined(RAJA_TEST_ENABLE_GPU)

GPU_TEST(Sort, basic_algorithm_device_Sort)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  testSorter(InsertionSortGPU{}, MaxN);
  testSorter(InsertionSortPairsGPU{}, MaxN);
  testSorter(ShellSortGPU{}, MaxN);
  testSorter(ShellSortPairsGPU{}, MaxN);
  testSorter(HeapSortGPU{}, MaxN);
  testSorter(HeapSortPairsGPU{}, MaxN);

  // intro_sort is implemented via recursion, so the device may
  // run out of stack space or perform poorly due to local memory usage
  testSorter(IntroSort{}, MaxN);
  testSorter(IntroSortPairs{}, MaxN);

  // merge_sort is not currently supported in device code due
  // to memory requirements
  // testSorter(MergeSort{}, MaxN);
  // testSorter(MergeSortPairs{}, MaxN);
}

#endif

