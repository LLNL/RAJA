//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for sort
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"
#include "type_helper.hpp"

#include <type_traits>
#include <algorithm>
#include <chrono>
#include <random>


template <typename T,
          typename Sorter>
void doSort(const T* orig, T* unsorted, RAJA::Index_type N, Sorter sorter)
{
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMemcpy(unsorted, orig, N*sizeof(T), cudaMemcpyDefault));
#else
  memcpy(unsorted, orig, N*sizeof(T));
#endif
  sorter(unsorted, unsorted+N);
}

template <typename T,
          typename KnownSorter,
          typename TestSorter>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    const T* orig,
    T* sorted,
    T* unsorted,
    RAJA::Index_type N,
    KnownSorter known_sorter,
    TestSorter test_sorter)
{
  doSort(orig,   sorted, N, known_sorter);
  doSort(orig, unsorted, N, test_sorter);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  for (RAJA::Index_type i = 0; i < N; i++) {
    if (unsorted[i] != sorted[i])
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ") "
             << unsorted[i] << " != " << sorted[i]
             << " (at index " << i << ")";
  }
  return ::testing::AssertionSuccess();
}


template <typename ExecPolicy,
          typename T>
void testSortIntegral(unsigned seed, RAJA::Index_type MaxN)
{
  std::mt19937 rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>((MaxN+1)/2, MaxN)(rng);

  T *orig = nullptr;
  T *sorted = nullptr;
  T *unsorted = nullptr;

  if (N > 0) {
    // initialize an array
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void **)&orig, sizeof(T) * N));
    cudaErrchk(cudaMallocManaged((void **)&sorted, sizeof(T) * N));
    cudaErrchk(cudaMallocManaged((void **)&unsorted, sizeof(T) * N));
    cudaErrchk(cudaDeviceSynchronize());
#else
    orig     = new T[N];
    sorted   = new T[N];
    unsorted = new T[N];
#endif

    // initialize orig to random values
    std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

    for (RAJA::Index_type i = 0; i < N; i++) {
      orig[i] = dist(rng);
    }
  }

  ASSERT_TRUE(testSort("sort default", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::sort(sorted, sorted_end);
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::sort<ExecPolicy>(unsorted, unsorted_end);
  }));

  // test ascending
  ASSERT_TRUE(testSort("sort ascending", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::sort(sorted, sorted_end, std::less<T>{});
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::sort<ExecPolicy>(unsorted, unsorted_end, RAJA::operators::less<T>{});
  }));

  // test descending
  ASSERT_TRUE(testSort("sort descending", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::sort(sorted, sorted_end, std::greater<T>{});
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::sort<ExecPolicy>(unsorted, unsorted_end, RAJA::operators::greater<T>{});
  }));


  // test default behavior
  ASSERT_TRUE(testSort("stable_sort default", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::stable_sort(sorted, sorted_end);
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::stable_sort<ExecPolicy>(unsorted, unsorted_end);
  }));

  // test ascending
  ASSERT_TRUE(testSort("stable_sort ascending", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::stable_sort(sorted, sorted_end, std::less<T>{});
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::stable_sort<ExecPolicy>(unsorted, unsorted_end, RAJA::operators::less<T>{});
  }));

  // test descending
  ASSERT_TRUE(testSort("stable-sort descending", seed, orig, sorted, unsorted, N,
      [](T* sorted, T* sorted_end) {
    std::stable_sort(sorted, sorted_end, std::greater<T>{});
  },  [](T* unsorted, T* unsorted_end) {
    RAJA::stable_sort<ExecPolicy>(unsorted, unsorted_end, RAJA::operators::greater<T>{});
  }));

  if (N > 0) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(orig));
    cudaErrchk(cudaFree(sorted));
    cudaErrchk(cudaFree(unsorted));
#else
    delete[] orig;
    delete[] sorted;
    delete[] unsorted;
#endif
  }
}


template <typename ExecPolicy>
void testSortSizesPol(unsigned seed, RAJA::Index_type MaxN)
{
  testSortIntegral<ExecPolicy, int>(seed, MaxN);
#if defined(TEST_EXHAUSTIVE)
  testSortIntegral<ExecPolicy, unsigned>(seed, MaxN);
  testSortIntegral<ExecPolicy, long long>(seed, MaxN);
  testSortIntegral<ExecPolicy, unsigned long long>(seed, MaxN);

  testSortIntegral<ExecPolicy, float>(seed, MaxN);
#endif
  testSortIntegral<ExecPolicy, double>(seed, MaxN);
}


template <typename ExecPolicy>
void testSortPol()
{
  unsigned seed = std::random_device{}();

  testSortSizesPol<ExecPolicy>(seed, 0);
  testSortSizesPol<ExecPolicy>(seed, 1);
  testSortSizesPol<ExecPolicy>(seed, 10);
  testSortSizesPol<ExecPolicy>(seed, 10000);
}
