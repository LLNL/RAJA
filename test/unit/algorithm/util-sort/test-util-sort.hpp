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
/// Header file containing Sorter classes for util sort tests
///

#ifndef __TEST_UTIL_SORT_HPP__
#define __TEST_UTIL_SORT_HPP__

#include "../test-sort.hpp"


using PolicySynchronizeCPU = PolicySynchronize<RAJA::loop_exec>;
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template < typename forone_policy >
using PolicySynchronizeGPU = PolicySynchronize<forone_equivalent_exec_policy<forone_policy>>;
#endif


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

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

template < typename forone_policy >
struct InsertionSortGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  InsertionSortGPU()
    : m_name(std::string("RAJA::insertion_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct InsertionSortPairsGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  InsertionSortPairsGPU()
    : m_name(std::string("RAJA::insertion_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
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
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct ShellSortGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  ShellSortGPU()
    : m_name(std::string("RAJA::shell_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct ShellSortPairsGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  ShellSortPairsGPU()
    : m_name(std::string("RAJA::shell_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
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
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct HeapSortGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  HeapSortGPU()
    : m_name(std::string("RAJA::heap_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct HeapSortPairsGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  HeapSortPairsGPU()
    : m_name(std::string("RAJA::heap_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
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
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct IntroSortGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  IntroSortGPU()
    : m_name(std::string("RAJA::intro_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct IntroSortPairsGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  IntroSortPairsGPU()
    : m_name(std::string("RAJA::intro_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
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
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct MergeSortGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  MergeSortGPU()
    : m_name(std::string("RAJA::merge_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct MergeSortPairsGPU
  : PolicySynchronizeGPU<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  MergeSortPairsGPU()
    : m_name(std::string("RAJA::merge_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
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
    forone_pol<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

#endif


using CpuInsertionSortSorters =
  camp::list<
              InsertionSort,
              InsertionSortPairs
            >;

using CpuShellSortSorters =
  camp::list<
              ShellSort,
              ShellSortPairs
            >;

using CpuHeapSortSorters =
  camp::list<
              HeapSort,
              HeapSortPairs
            >;

using CpuIntroSortSorters =
  camp::list<
              IntroSort,
              IntroSortPairs
            >;

using CpuMergeSortSorters =
  camp::list<
              MergeSort,
              MergeSortPairs
            >;

#if defined(RAJA_ENABLE_CUDA)

using CudaInsertionSortSorters =
  camp::list<
              InsertionSortGPU<forone_cuda>,
              InsertionSortPairsGPU<forone_cuda>
            >;

using CudaShellSortSorters =
  camp::list<
              ShellSortGPU<forone_cuda>,
              ShellSortPairsGPU<forone_cuda>
            >;

using CudaHeapSortSorters =
  camp::list<
              HeapSortGPU<forone_cuda>,
              HeapSortPairsGPU<forone_cuda>
            >;

using CudaIntroSortSorters =
  camp::list<
              IntroSortGPU<forone_cuda>,
              IntroSortPairsGPU<forone_cuda>
            >;

using CudaMergeSortSorters =
  camp::list<
              MergeSortGPU<forone_cuda>,
              MergeSortPairsGPU<forone_cuda>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipInsertionSortSorters =
  camp::list<
              InsertionSortGPU<forone_hip>,
              InsertionSortPairsGPU<forone_hip>
            >;

using HipShellSortSorters =
  camp::list<
              ShellSortGPU<forone_hip>,
              ShellSortPairsGPU<forone_hip>
            >;

using HipHeapSortSorters =
  camp::list<
              HeapSortGPU<forone_hip>,
              HeapSortPairsGPU<forone_hip>
            >;

using HipIntroSortSorters =
  camp::list<
              IntroSortGPU<forone_hip>,
              IntroSortPairsGPU<forone_hip>
            >;

using HipMergeSortSorters =
  camp::list<
              MergeSortGPU<forone_hip>,
              MergeSortPairsGPU<forone_hip>
            >;

#endif

#endif //__TEST_UTIL_SORT_HPP__

