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

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <random>


template <typename T,
          typename Sorter>
void doSort(const T* orig, T* sorted, RAJA::Index_type N, Sorter sorter)
{
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMemcpy(sorted, orig, N*sizeof(T), cudaMemcpyDefault));
#else
  memcpy(sorted, orig, N*sizeof(T));
#endif
  sorter(sorted, sorted+N);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
}

template <typename K,
          typename V,
          typename Sorter>
void doSortPairs(const K* orig_keys, K* sorted_keys,
                 const V* orig_vals, V* sorted_vals,
                 RAJA::Index_type N, Sorter sorter)
{
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMemcpy(sorted_keys, orig_keys, N*sizeof(K), cudaMemcpyDefault));
  cudaErrchk(cudaMemcpy(sorted_vals, orig_vals, N*sizeof(V), cudaMemcpyDefault));
#else
  memcpy(sorted_keys, orig_keys, N*sizeof(K));
  memcpy(sorted_vals, orig_vals, N*sizeof(V));
#endif
  sorter(sorted_keys, sorted_keys+N, sorted_vals);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
}


template <typename T,
          typename Compare,
          typename TestSorter>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    RAJA::Index_type N,
    Compare comp,
    const T* orig,
    T* sorted,
    TestSorter test_sorter)
{
  doSort(orig, sorted, N, test_sorter);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  // make map of keys to keys
  using val_map = std::unordered_multiset<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(orig[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(orig[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(orig[i]);
  }
  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(sorted[i], sorted[i-1]))
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << sorted[i-1] << ", " << sorted[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(sorted[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << sorted[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(sorted[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << sorted[i]
             << " (at index " << i << ")";
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T,
          typename Compare,
          typename TestSorter>
::testing::AssertionResult testStableSort(
    const char* test_name,
    const unsigned seed,
    RAJA::Index_type N,
    Compare comp,
    const T* orig,
    T* sorted,
    TestSorter test_sorter)
{
  doSort(orig, sorted, N, test_sorter);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  // make map of keys to keys
  using val_map = std::list<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(orig[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(orig[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(orig[i]);
  }
  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(sorted[i], sorted[i-1]))
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << sorted[i-1] << ", " << sorted[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(sorted[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << sorted[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != sorted[i])
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << sorted[i]
             << " (at index " << i << ")";
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}


template <typename K,
          typename V,
          typename Compare,
          typename TestSorter>
::testing::AssertionResult testSortPairs(
    const char* test_name,
    const unsigned seed,
    RAJA::Index_type N,
    Compare comp,
    const K* orig_keys,
    K* sorted_keys,
    const V* orig_vals,
    V* sorted_vals,
    TestSorter test_sorter)
{
  doSortPairs(orig_keys,   sorted_keys,
              orig_vals,   sorted_vals,
              N, test_sorter);
  // make map of keys to vals
  using val_map = std::unordered_multiset<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(orig_vals[i]);
  }
  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(sorted_keys[i], sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " keys " << sorted_keys[i-1] << ", " << sorted_keys[i] << " out of order"
             << " vals " << sorted_vals[i-1] << ", " << sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << sorted_keys[i]
             << " val " << sorted_vals[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(sorted_vals[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << " key " << sorted_keys[i]
             << " val " << sorted_vals[i]
             << " (at index " << i << ")";
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename K,
          typename V,
          typename Compare,
          typename TestSorter>
::testing::AssertionResult testStableSortPairs(
    const char* test_name,
    const unsigned seed,
    RAJA::Index_type N,
    Compare comp,
    const K* orig_keys,
    K* sorted_keys,
    const V* orig_vals,
    V* sorted_vals,
    TestSorter test_sorter)
{
  doSortPairs(orig_keys,   sorted_keys,
              orig_vals,   sorted_vals,
              N, test_sorter);
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif
  // make map of keys to vals
  using val_map = std::list<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(orig_vals[i]);
  }
  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(sorted_keys[i], sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << " keys " << sorted_keys[i-1] << ", " << sorted_keys[i]
             << " vals " << sorted_vals[i-1] << ", " << sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << sorted_keys[i]
             << " val " << sorted_vals[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != sorted_vals[i])
      return ::testing::AssertionFailure()
             << test_name << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << " key " << sorted_keys[i]
             << " val " << sorted_vals[i]
             << " (at index " << i << ")";
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename ExecPolicy,
          typename K>
void testSorts(unsigned seed, RAJA::Index_type MaxN)
{
  std::mt19937 rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>((MaxN+1)/2, MaxN)(rng);

  K *orig = nullptr;
  K *sorted = nullptr;

  if (N > 0) {
    // initialize an array
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void **)&orig, sizeof(K) * N));
    cudaErrchk(cudaMallocManaged((void **)&sorted, sizeof(K) * N));
    cudaErrchk(cudaDeviceSynchronize());
#else
    orig     = new K[N];
    sorted   = new K[N];
#endif

    // initialize orig to random values
    std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

    for (RAJA::Index_type i = 0; i < N; i++) {
      orig[i] = dist(rng);
    }
  }


  ASSERT_TRUE(testSort("sort default", seed, N, std::less<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::sort<ExecPolicy>(sorted, sorted_end);
  }));

  // test ascending
  ASSERT_TRUE(testSort("sort ascending", seed, N, std::less<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::sort<ExecPolicy>(sorted, sorted_end, RAJA::operators::less<K>{});
  }));

  // test descending
  ASSERT_TRUE(testSort("sort descending", seed, N, std::greater<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::sort<ExecPolicy>(sorted, sorted_end, RAJA::operators::greater<K>{});
  }));


  // test default behavior
  ASSERT_TRUE(testStableSort("stable_sort default", seed, N, std::less<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::stable_sort<ExecPolicy>(sorted, sorted_end);
  }));

  // test ascending
  ASSERT_TRUE(testStableSort("stable_sort ascending", seed, N, std::less<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::stable_sort<ExecPolicy>(sorted, sorted_end, RAJA::operators::less<K>{});
  }));

  // test descending
  ASSERT_TRUE(testStableSort("stable_sort descending", seed, N, std::greater<K>{},
      orig, sorted,
      [](K* sorted, K* sorted_end) {
    RAJA::stable_sort<ExecPolicy>(sorted, sorted_end, RAJA::operators::greater<K>{});
  }));


  if (N > 0) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(orig));
    cudaErrchk(cudaFree(sorted));
#else
    delete[] orig;
    delete[] sorted;
#endif
  }
}

template <typename ExecPolicy,
          typename K,
          typename V>
void testSortsPairs(unsigned seed, RAJA::Index_type MaxN)
{
  std::mt19937 rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>((MaxN+1)/2, MaxN)(rng);

  K *orig_keys = nullptr;
  K *sorted_keys = nullptr;
  V *orig_vals = nullptr;
  V *sorted_vals = nullptr;

  if (N > 0) {
    // initialize an array
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void **)&orig_keys, sizeof(K) * N));
    cudaErrchk(cudaMallocManaged((void **)&sorted_keys, sizeof(K) * N));
    cudaErrchk(cudaMallocManaged((void **)&orig_vals, sizeof(V) * N));
    cudaErrchk(cudaMallocManaged((void **)&sorted_vals, sizeof(V) * N));
    cudaErrchk(cudaDeviceSynchronize());
#else
    orig_keys     = new K[N];
    sorted_keys   = new K[N];
    orig_vals     = new V[N];
    sorted_vals   = new V[N];
#endif

    // initialize orig to random values
    std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

    for (RAJA::Index_type i = 0; i < N; i++) {
      orig_keys[i] = dist(rng);
    }
    for (RAJA::Index_type i = 0; i < N; i++) {
      orig_vals[i] = dist(rng);
    }
  }


  ASSERT_TRUE(testSortPairs("sort_pairs default", seed, N, std::less<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals);
  }));

  // test ascending
  ASSERT_TRUE(testSortPairs("sort_pairs ascending", seed, N, std::less<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals, RAJA::operators::less<K>{});
  }));

  // test descending
  ASSERT_TRUE(testSortPairs("sort_pairs descending", seed, N, std::greater<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals, RAJA::operators::greater<K>{});
  }));


  // test default behavior
  ASSERT_TRUE(testStableSortPairs("stable_sort_pairs default", seed, N, std::less<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::stable_sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals);
  }));

  // test ascending
  ASSERT_TRUE(testStableSortPairs("stable_sort_pairs ascending", seed, N, std::less<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::stable_sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals, RAJA::operators::less<K>{});
  }));

  // test descending
  ASSERT_TRUE(testStableSortPairs("stable_sort_pairs descending", seed, N, std::greater<K>{},
      orig_keys, sorted_keys, orig_vals, sorted_vals,
      [](K* sorted_keys, K* sorted_keys_end, V* sorted_vals) {
    RAJA::stable_sort_pairs<ExecPolicy>(sorted_keys, sorted_keys_end, sorted_vals, RAJA::operators::greater<K>{});
  }));


  if (N > 0) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(orig_keys));
    cudaErrchk(cudaFree(sorted_keys));
    cudaErrchk(cudaFree(orig_vals));
    cudaErrchk(cudaFree(sorted_vals));
#else
    delete[] orig_keys;
    delete[] sorted_keys;
    delete[] orig_vals;
    delete[] sorted_vals;
#endif
  }
}



template <typename ExecPolicy>
void testSortSizesPol(unsigned seed, RAJA::Index_type MaxN)
{
  testSorts<ExecPolicy, int>(seed, MaxN);
#if defined(TEST_EXHAUSTIVE)
  testSorts<ExecPolicy, unsigned>(seed, MaxN);
  testSorts<ExecPolicy, long long>(seed, MaxN);
  testSorts<ExecPolicy, unsigned long long>(seed, MaxN);

  testSorts<ExecPolicy, float>(seed, MaxN);
#endif
  testSorts<ExecPolicy, double>(seed, MaxN);
}

template <typename ExecPolicy>
void testSortPairsSizesPol(unsigned seed, RAJA::Index_type MaxN)
{
  testSortsPairs<ExecPolicy, int,                int>(seed, MaxN);
#if defined(TEST_EXHAUSTIVE)
  testSortsPairs<ExecPolicy, unsigned,           int>(seed, MaxN);
  testSortsPairs<ExecPolicy, long long,          int>(seed, MaxN);
  testSortsPairs<ExecPolicy, unsigned long long, int>(seed, MaxN);

  testSortsPairs<ExecPolicy, float,              int>(seed, MaxN);
#endif
  testSortsPairs<ExecPolicy, double,             int>(seed, MaxN);
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

template <typename ExecPolicy>
void testSortPairsPol()
{
  unsigned seed = std::random_device{}();

  testSortPairsSizesPol<ExecPolicy>(seed, 0);
  testSortPairsSizesPol<ExecPolicy>(seed, 1);
  testSortPairsSizesPol<ExecPolicy>(seed, 10);
  testSortPairsSizesPol<ExecPolicy>(seed, 10000);
}
