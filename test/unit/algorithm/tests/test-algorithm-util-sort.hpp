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

#ifndef __TEST_ALGORITHM_UTIL_SORT_HPP__
#define __TEST_ALGORITHM_UTIL_SORT_HPP__

#include "test-algorithm-sort-utils.hpp"


template < typename forone_policy >
using ForoneSynchronize = PolicySynchronize<forone_equivalent_exec_policy<forone_policy>>;


template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct InsertionSort;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct InsertionSortPairs;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct ShellSort;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct ShellSortPairs;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct HeapSort;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct HeapSortPairs;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct IntroSort;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct IntroSortPairs;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct MergeSort;

template < typename forone_policy, typename platform = forone_platform<forone_policy> >
struct MergeSortPairs;


template < typename forone_policy >
struct InsertionSort<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct InsertionSortPairs<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct ShellSort<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct ShellSortPairs<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct HeapSort<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct HeapSortPairs<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct IntroSort<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct IntroSortPairs<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct MergeSort<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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

template < typename forone_policy >
struct MergeSortPairs<forone_policy, RunOnHost>
  : ForoneSynchronize<forone_policy>
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
struct InsertionSort<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  InsertionSort()
    : m_name(std::string("RAJA::insertion_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct InsertionSortPairs<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  InsertionSortPairs()
    : m_name(std::string("RAJA::insertion_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
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
    forone<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::insertion_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct ShellSort<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  ShellSort()
    : m_name(std::string("RAJA::shell_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct ShellSortPairs<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  ShellSortPairs()
    : m_name(std::string("RAJA::shell_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
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
    forone<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::shell_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct HeapSort<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  HeapSort()
    : m_name(std::string("RAJA::heap_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct HeapSortPairs<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  HeapSortPairs()
    : m_name(std::string("RAJA::heap_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
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
    forone<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::heap_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct IntroSort<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  IntroSort()
    : m_name(std::string("RAJA::intro_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct IntroSortPairs<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  IntroSortPairs()
    : m_name(std::string("RAJA::intro_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
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
    forone<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::intro_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename forone_policy >
struct MergeSort<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  MergeSort()
    : m_name(std::string("RAJA::merge_sort<") + forone_policy_info<forone_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter >
  void operator()(Iter begin, Iter end)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(begin, end);
    });
  }

  template < typename Iter, typename Compare >
  void operator()(Iter begin, Iter end, Compare comp)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(begin, end, comp);
    });
  }
};

template < typename forone_policy >
struct MergeSortPairs<forone_policy, RunOnDevice>
  : ForoneSynchronize<forone_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  MergeSortPairs()
    : m_name(std::string("RAJA::merge_sort<") + forone_policy_info<forone_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter >
  void operator()(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin)
  {
    forone<forone_policy>( [=] RAJA_DEVICE() {
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
    forone<forone_policy>( [=] RAJA_DEVICE() {
      auto begin = RAJA::zip(keys_begin, vals_begin);
      auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
      using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
      RAJA::merge_sort(begin, end, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

#endif


using SequentialInsertionSortSorters =
  camp::list<
              InsertionSort<forone_seq>,
              InsertionSortPairs<forone_seq>
            >;

using SequentialShellSortSorters =
  camp::list<
              ShellSort<forone_seq>,
              ShellSortPairs<forone_seq>
            >;

using SequentialHeapSortSorters =
  camp::list<
              HeapSort<forone_seq>,
              HeapSortPairs<forone_seq>
            >;

using SequentialIntroSortSorters =
  camp::list<
              IntroSort<forone_seq>,
              IntroSortPairs<forone_seq>
            >;

using SequentialMergeSortSorters =
  camp::list<
              MergeSort<forone_seq>,
              MergeSortPairs<forone_seq>
            >;

#if defined(RAJA_ENABLE_CUDA)

using CudaInsertionSortSorters =
  camp::list<
              InsertionSort<forone_cuda>,
              InsertionSortPairs<forone_cuda>
            >;

using CudaShellSortSorters =
  camp::list<
              ShellSort<forone_cuda>,
              ShellSortPairs<forone_cuda>
            >;

using CudaHeapSortSorters =
  camp::list<
              HeapSort<forone_cuda>,
              HeapSortPairs<forone_cuda>
            >;

using CudaIntroSortSorters =
  camp::list<
              IntroSort<forone_cuda>,
              IntroSortPairs<forone_cuda>
            >;

using CudaMergeSortSorters =
  camp::list<
              MergeSort<forone_cuda>,
              MergeSortPairs<forone_cuda>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipInsertionSortSorters =
  camp::list<
              InsertionSort<forone_hip>,
              InsertionSortPairs<forone_hip>
            >;

using HipShellSortSorters =
  camp::list<
              ShellSort<forone_hip>,
              ShellSortPairs<forone_hip>
            >;

using HipHeapSortSorters =
  camp::list<
              HeapSort<forone_hip>,
              HeapSortPairs<forone_hip>
            >;

using HipIntroSortSorters =
  camp::list<
              IntroSort<forone_hip>,
              IntroSortPairs<forone_hip>
            >;

using HipMergeSortSorters =
  camp::list<
              MergeSort<forone_hip>,
              MergeSortPairs<forone_hip>
            >;

#endif

#endif //__TEST_ALGORITHM_UTIL_SORT_HPP__

