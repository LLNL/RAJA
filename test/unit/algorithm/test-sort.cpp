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

TEST(Sort, basic_insertion_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(InsertionSort{}, MaxN);
  testSorter(InsertionSortPairs{}, MaxN);
}

TEST(Sort, basic_shell_Sort)
{
  testSorter(ShellSort{});
  testSorter(ShellSortPairs{});
}

TEST(Sort, basic_heap_Sort)
{
  testSorter(HeapSort{});
  testSorter(HeapSortPairs{});
}

TEST(Sort, basic_intro_Sort)
{
  testSorter(IntroSort{});
  testSorter(IntroSortPairs{});
}

TEST(Sort, basic_merge_Sort)
{
  testSorter(MergeSort{});
  testSorter(MergeSortPairs{});
}

TEST(Sort, basic_loop_Sort)
{
  testSorter(PolicySort<RAJA::loop_exec>{"loop"});
  testSorter(PolicySortPairs<RAJA::loop_exec>{"loop"});
}

TEST(Sort, basic_loop_StableSort)
{
  testSorter(PolicyStableSort<RAJA::loop_exec>{"loop"});
  testSorter(PolicyStableSortPairs<RAJA::loop_exec>{"loop"});
}

TEST(Sort, basic_seq_Sort)
{
  testSorter(PolicySort<RAJA::seq_exec>{"seq"});
  testSorter(PolicySortPairs<RAJA::seq_exec>{"seq"});
}

TEST(Sort, basic_seq_StableSort)
{
  testSorter(PolicyStableSort<RAJA::seq_exec>{"seq"});
  testSorter(PolicyStableSortPairs<RAJA::seq_exec>{"seq"});
}

#if defined(RAJA_ENABLE_OPENMP)

TEST(Sort, basic_OpenMP_Sort)
{
  testSorter(PolicySort<RAJA::omp_parallel_for_exec>{"omp"});
  // testSorter(PolicySortPairs<RAJA::omp_parallel_for_exec>{"omp"});
}

TEST(Sort, basic_OpenMP_StableSort)
{
  testSorter(PolicyStableSort<RAJA::omp_parallel_for_exec>{"omp"});
  // testSorter(PolicyStableSortPairs<RAJA::omp_parallel_for_exec>{"omp"});
}

#endif

#if defined(RAJA_ENABLE_TBB)

TEST(Sort, basic_TBB_Sort)
{
  testSorter(PolicySort<RAJA::tbb_for_exec>{"tbb"});
  // testSorter(PolicySortPairs<RAJA::tbb_for_exec>{"tbb"});
}

TEST(Sort, basic_TBB_StableSort)
{
  testSorter(PolicyStableSort<RAJA::tbb_for_exec>{"tbb"});
  // testSorter(PolicyStableSortPairs<RAJA::tbb_for_exec>{"tbb"});
}

#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Sort, basic_CUDA_Sort)
{
  testSorter(PolicySort<RAJA::cuda_exec<256>>{"cuda"});
  testSorter(PolicySortPairs<RAJA::cuda_exec<256>>{"cuda"});
}

GPU_TEST(Sort, basic_CUDA_StableSort)
{
  testSorter(PolicyStableSort<RAJA::cuda_exec<256>>{"cuda"});
  testSorter(PolicyStableSortPairs<RAJA::cuda_exec<256>>{"cuda"});
}

GPU_TEST(Sort, basic_CUDA_device_insertion_Sort)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  testSorter(InsertionSortGPU<forone_cuda>{}, MaxN);
  testSorter(InsertionSortPairsGPU<forone_cuda>{}, MaxN);
}

GPU_TEST(Sort, basic_CUDA_device_shell_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(ShellSortGPU<forone_cuda>{}, MaxN);
  testSorter(ShellSortPairsGPU<forone_cuda>{}, MaxN);
}

GPU_TEST(Sort, basic_CUDA_device_heap_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(HeapSortGPU<forone_cuda>{}, MaxN);
  testSorter(HeapSortPairsGPU<forone_cuda>{}, MaxN);
}

#if 0
GPU_TEST(Sort, basic_CUDA_device_intro_Sort)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  // intro_sort is implemented via recursion, so the device may
  // run out of stack space or perform poorly due to local memory usage
  // or fail to link with hip
  testSorter(IntroSortGPU<forone_cuda>{}, MaxN);
  testSorter(IntroSortPairsGPU<forone_cuda>{}, MaxN);
}
#endif

#if 0
GPU_TEST(Sort, basic_CUDA_device_merge_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  // merge_sort is not currently supported in device code due
  // to memory requirements
  testSorter(MergeSortGPU<forone_cuda>{}, MaxN);
  testSorter(MergeSortPairsGPU<forone_cuda>{}, MaxN);
}
#endif

#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Sort, basic_HIP_Sort)
{
  testSorter(PolicySort<RAJA::hip_exec<256>>{"hip"});
  testSorter(PolicySortPairs<RAJA::hip_exec<256>>{"hip"});
}

GPU_TEST(Sort, basic_HIP_StableSort)
{
  testSorter(PolicyStableSort<RAJA::hip_exec<256>>{"hip"});
  testSorter(PolicyStableSortPairs<RAJA::hip_exec<256>>{"hip"});
}

GPU_TEST(Sort, basic_HIP_device_insertion_Sort)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  testSorter(InsertionSortGPU<forone_hip>{}, MaxN);
  testSorter(InsertionSortPairsGPU<forone_hip>{}, MaxN);
}

GPU_TEST(Sort, basic_HIP_device_shell_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(ShellSortGPU<forone_hip>{}, MaxN);
  testSorter(ShellSortPairsGPU<forone_hip>{}, MaxN);
}

GPU_TEST(Sort, basic_HIP_device_heap_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  testSorter(HeapSortGPU<forone_hip>{}, MaxN);
  testSorter(HeapSortPairsGPU<forone_hip>{}, MaxN);
}

#if 0
GPU_TEST(Sort, basic_HIP_device_intro_Sort)
{
  RAJA::Index_type MaxN = 100; // limit MaxN to decrease runtime
  // intro_sort is implemented via recursion, so the device may
  // run out of stack space or perform poorly due to local memory usage
  // or fail to link with hip
  testSorter(IntroSortGPU<forone_hip>{}, MaxN);
  testSorter(IntroSortPairsGPU<forone_hip>{}, MaxN);
}
#endif

#if 0
GPU_TEST(Sort, basic_HIP_device_merge_Sort)
{
  RAJA::Index_type MaxN = 1000; // limit MaxN to decrease runtime
  // merge_sort is not currently supported in device code due
  // to memory requirements
  testSorter(MergeSortGPU<forone_hip>{}, MaxN);
  testSorter(MergeSortPairsGPU<forone_hip>{}, MaxN);
}
#endif

#endif

