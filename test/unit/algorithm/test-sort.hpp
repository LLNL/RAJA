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
#include "RAJA_unit_forone.hpp"

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <random>


struct unstable_sort_tag { };
struct stable_sort_tag { };

struct sort_interface_tag { };
struct sort_pairs_interface_tag { };

struct sort_default_interface_tag { };
struct sort_comp_interface_tag { };

// synchronize based on a RAJA execution policy
template < typename policy >
struct PolicySynchronize
{
  void synchronize()
  {
    // no synchronization needed
  }
};

#if defined(RAJA_ENABLE_CUDA)
// partial specializatoin for cuda_exec
template < size_t BLOCK_SIZE, bool Async >
struct PolicySynchronize<RAJA::cuda_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async) { RAJA::synchronize<RAJA::cuda_synchronize>(); }
  }
};
#endif

#if defined(RAJA_ENABLE_HIP)
// partial specializatoin for hip_exec
template < size_t BLOCK_SIZE, bool Async >
struct PolicySynchronize<RAJA::hip_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async) { RAJA::synchronize<RAJA::hip_synchronize>(); }
  }
};
#endif

using PolicySynchronizeCPU = PolicySynchronize<RAJA::loop_exec>;
#ifdef RAJA_TEST_ENABLE_GPU
using PolicySynchronizeGPU = PolicySynchronize<forone_equivalent_exec_policy>;
#endif


template < typename policy >
struct PolicySort
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::sort<policy>";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::sort<policy>(std::forward<Args>(args)...);
  }
};

template < typename policy >
struct PolicyStableSort
  : PolicySynchronize<policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::stable_sort<policy>";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::stable_sort<policy>(std::forward<Args>(args)...);
  }
};

template < typename policy >
struct PolicySortPairs
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::sort_pairs<policy>";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::sort_pairs<policy>(std::forward<Args>(args)...);
  }
};

template < typename policy >
struct PolicyStableSortPairs
  : PolicySynchronize<policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::stable_sort_pairs<policy>";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::stable_sort_pairs<policy>(std::forward<Args>(args)...);
  }
};

struct InsertionSort
  : PolicySynchronizeCPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::insertion_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::insertion_sort(std::forward<Args>(args)...);
  }
};

struct InsertionSortPairs
  : PolicySynchronizeCPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::insertion_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }
};

struct ShellSort
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::shell_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::shell_sort(std::forward<Args>(args)...);
  }
};

struct ShellSortPairs
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::shell_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }
};

struct HeapSort
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::heap_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::heap_sort(std::forward<Args>(args)...);
  }
};

struct HeapSortPairs
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::heap_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }
};

struct IntroSort
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::intro_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::intro_sort(std::forward<Args>(args)...);
  }
};

struct IntroSortPairs
  : PolicySynchronizeCPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::intro_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }
};

struct MergeSort
  : PolicySynchronizeCPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::merge_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::merge_sort(std::forward<Args>(args)...);
  }
};

struct MergeSortPairs
  : PolicySynchronizeCPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::merge_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    auto begin = RAJA::zip(keys_begin, vals_begin);
    auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
    using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
    RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
  }
};


#ifdef RAJA_TEST_ENABLE_GPU

struct InsertionSortGPU
  : PolicySynchronizeGPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::insertion_sort";
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::insertion_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::insertion_sort(begin, end, comp);
    });
  }
};

struct InsertionSortPairsGPU
  : PolicySynchronizeGPU
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::insertion_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

struct ShellSortGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::shell_sort";
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::shell_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::shell_sort(begin, end, comp);
    });
  }
};

struct ShellSortPairsGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::shell_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
    RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

struct HeapSortGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::heap_sort";
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::heap_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::heap_sort(begin, end, comp);
    });
  }
};

struct HeapSortPairsGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::heap_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
      RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

struct IntroSortGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::intro_sort";
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::intro_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::intro_sort(begin, end, comp);
    });
  }
};

struct IntroSortPairsGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::intro_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
      RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

struct MergeSortGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  const char* name()
  {
    return "RAJA::merge_sort";
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::merge_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      RAJA::merge_sort(begin, end, comp);
    });
  }
};

struct MergeSortPairsGPU
  : PolicySynchronizeGPU
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  const char* name()
  {
    return "RAJA::merge_sort[pairs]";
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::operators::less<RAJA::detail::IterRef<KeyIter>> comp{};
      RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }

  template < typename KeyIter, typename ValIter, typename Compare >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin, Compare comp)
  {
    forone_gpu( RAJA_TEST_DEVICE_LAMBDA() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

#endif


template <typename pairs_category,
          typename K,
          typename V = RAJA::Index_type>
struct SortData;

template <typename K, typename V>
struct SortData<sort_interface_tag, K, V>
{
  K* orig_keys = nullptr;
  K* sorted_keys = nullptr;

  template < typename RandomGenerator >
  SortData(size_t N, RandomGenerator gen_random)
  {
    if (N > 0) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaMallocManaged((void **)&orig_keys, sizeof(K) * N));
      cudaErrchk(cudaMallocManaged((void **)&sorted_keys, sizeof(K) * N));
#else
      orig_keys   = new K[N];
      sorted_keys = new K[N];
#endif
    }
    cudaErrchk(cudaDeviceSynchronize());

    for (size_t i = 0; i < N; i++) {
      orig_keys[i] = gen_random();
    }
  }

  void copy_data(size_t N) const
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMemcpy(sorted_keys, orig_keys, N*sizeof(K), cudaMemcpyDefault));
#else
    memcpy(sorted_keys, orig_keys, N*sizeof(K));
#endif
  }

  SortData(SortData const&) = delete;
  SortData& operator=(SortData const&) = delete;

  ~SortData()
  {
    if (orig_keys != nullptr) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaFree(orig_keys));
      cudaErrchk(cudaFree(sorted_keys));
#else
      delete[] orig_keys;
      delete[] sorted_keys;
#endif
    }
  }
};


template <typename K, typename V>
struct SortData<sort_pairs_interface_tag, K, V> : SortData<sort_interface_tag, K, V>
{
  using base = SortData<sort_interface_tag, K, V>;

  V* orig_vals = nullptr;
  V* sorted_vals = nullptr;

  template < typename RandomGenerator >
  SortData(size_t N, RandomGenerator gen_random)
    : base(N, gen_random)
  {
    if (N > 0) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaMallocManaged((void **)&orig_vals, sizeof(V) * N));
      cudaErrchk(cudaMallocManaged((void **)&sorted_vals, sizeof(V) * N));
#else
      orig_vals   = new V[N];
      sorted_vals = new V[N];
#endif
    }
    cudaErrchk(cudaDeviceSynchronize());

    for (size_t i = 0; i < N; i++) {
      orig_vals[i] = gen_random();
    }
  }

  void copy_data(size_t N) const
  {
    base::copy_data(N);
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMemcpy(sorted_vals, orig_vals, N*sizeof(V), cudaMemcpyDefault));
#else
    memcpy(sorted_vals, orig_vals, N*sizeof(V));
#endif
  }

  SortData(SortData const&) = delete;
  SortData& operator=(SortData const&) = delete;

  ~SortData()
  {
    if (orig_vals != nullptr) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaFree(orig_vals));
      cudaErrchk(cudaFree(sorted_vals));
#else
      delete[] orig_vals;
      delete[] sorted_vals;
#endif
    }
  }
};


template <typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<sort_interface_tag, T> const& data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_interface_tag, sort_default_interface_tag)
{
  data.copy_data(N);
  sorter(data.sorted_keys, data.sorted_keys+N);
  sorter.synchronize();
}

template <typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<sort_interface_tag, T> const& data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_interface_tag, sort_comp_interface_tag)
{
  data.copy_data(N);
  sorter(data.sorted_keys, data.sorted_keys+N, comp);
  sorter.synchronize();
}

template <typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<sort_pairs_interface_tag, K, V> const& data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_pairs_interface_tag, sort_default_interface_tag)
{
  data.copy_data(N);
  sorter(data.sorted_keys, data.sorted_keys+N, data.sorted_vals);
  sorter.synchronize();
}

template <typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<sort_pairs_interface_tag, K, V> const& data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_pairs_interface_tag, sort_comp_interface_tag)
{
  data.copy_data(N);
  sorter(data.sorted_keys, data.sorted_keys+N, data.sorted_vals, comp);
  sorter.synchronize();
}


template <typename T,
          typename Compare,
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<sort_interface_tag, T> const& data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, unstable_sort_tag, sort_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to keys
  using val_map = std::unordered_multiset<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(data.orig_keys[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(data.orig_keys[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(data.sorted_keys[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(data.sorted_keys[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << data.sorted_keys[i]
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
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<sort_interface_tag, T> const& data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, stable_sort_tag, sort_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to keys
  using val_map = std::list<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(data.orig_keys[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(data.orig_keys[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(data.sorted_keys[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != data.sorted_keys[i])
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << data.sorted_keys[i]
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
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<sort_pairs_interface_tag, K, V> const& data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, unstable_sort_tag, sort_pairs_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to vals
  using val_map = std::unordered_multiset<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(data.orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(data.orig_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " keys " << data.sorted_keys[i-1] << ", " << data.sorted_keys[i] << " out of order"
             << " vals " << data.sorted_vals[i-1] << ", " << data.sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(data.sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(data.sorted_vals[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
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
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<sort_pairs_interface_tag, K, V> const& data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, stable_sort_tag, sort_pairs_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to vals
  using val_map = std::list<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(data.orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(data.orig_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << " keys " << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " vals " << data.sorted_vals[i-1] << ", " << data.sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(data.sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != data.sorted_vals[i])
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename K,
          typename Sorter>
void testSorterInterfaces(unsigned seed, RAJA::Index_type MaxN, Sorter sorter)
{
  using stability_category = typename Sorter::sort_category ;
  using pairs_category     = typename Sorter::sort_interface ;
  using no_comparator      = sort_default_interface_tag;
  using use_comparator     = sort_comp_interface_tag;

  std::mt19937 rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>((MaxN+1)/2, MaxN)(rng);
  std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

  SortData<pairs_category, K> data(N, [&](){ return dist(rng); });

  ASSERT_TRUE(testSort("default", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, no_comparator{}));
  ASSERT_TRUE(testSort("ascending", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, use_comparator{}));
  ASSERT_TRUE(testSort("descending", seed, data, N, RAJA::operators::greater<K>{},
      sorter, stability_category{}, pairs_category{}, use_comparator{}));
}

template <typename Sorter>
void testSorterSizes(unsigned seed, RAJA::Index_type MaxN, Sorter sorter)
{
  testSorterInterfaces<int>(seed, MaxN, sorter);
#if defined(TEST_EXHAUSTIVE)
  testSorterInterfaces<unsigned>(seed, MaxN), sorter;
  testSorterInterfaces<long long>(seed, MaxN, sorter);
  testSorterInterfaces<unsigned long long>(seed, MaxN, sorter);

  testSorterInterfaces<float>(seed, MaxN, sorter);
#endif
  testSorterInterfaces<double>(seed, MaxN, sorter);
}

inline unsigned get_random_seed()
{
  static unsigned seed = std::random_device{}();
  return seed;
}

template <typename Sorter>
void testSorter(Sorter sorter, RAJA::Index_type MaxN = 10000)
{
  unsigned seed = get_random_seed();

  testSorterSizes(seed, 0, sorter);
  for (RAJA::Index_type n = 1; n <= MaxN; n *= 10) {
    testSorterSizes(seed, n, sorter);
  }
}
