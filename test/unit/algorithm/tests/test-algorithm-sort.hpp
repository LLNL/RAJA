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
/// Header file containing Sorter classes for sort tests
///

#ifndef __TEST_UNIT_ALGORITHM_SORT_HPP__
#define __TEST_UNIT_ALGORITHM_SORT_HPP__

#include "test-algorithm-sort-utils.hpp"

template < typename policy >
struct PolicySort
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;

  std::string m_name;

  PolicySort()
    : m_name("RAJA::sort<unknown>")
  { }

  PolicySort(std::string const& policy_name)
    : m_name(std::string("RAJA::sort<") + policy_name + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Iter, typename... Args >
  void operator()(Iter begin, Iter end, Args&&... args)
  {
    using std::distance;
    auto N = distance(begin, end);
    RAJA::sort<policy>(
        RAJA::make_span(begin, N),
        std::forward<Args>(args)...);
  }
};

template < typename policy >
struct PolicySortPairs
  : PolicySynchronize<policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;

  std::string m_name;

  PolicySortPairs()
    : m_name("RAJA::sort<unknown>[pairs]")
  { }

  PolicySortPairs(std::string const& policy_name)
    : m_name(std::string("RAJA::sort<") + policy_name + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyIter, typename ValIter, typename... Args >
  void operator()(KeyIter keys_begin, KeyIter keys_end,
                  ValIter vals_begin,
                  Args&&... args)
  {
    using std::distance;
    auto N = distance(keys_begin, keys_end);
    RAJA::sort_pairs<policy>(
        RAJA::make_span(keys_begin, N),
        RAJA::make_span(vals_begin, N),
        std::forward<Args>(args)...);
  }
};


using SequentialSortSorters =
  camp::list<
              PolicySort<RAJA::loop_exec>,
              PolicySortPairs<RAJA::loop_exec>,
              PolicySort<RAJA::seq_exec>,
              PolicySortPairs<RAJA::seq_exec>
            >;

#if defined(RAJA_ENABLE_OPENMP)

using OpenMPSortSorters =
  camp::list<
              PolicySort<RAJA::omp_parallel_for_exec>,
              PolicySortPairs<RAJA::omp_parallel_for_exec>
            >;

#endif

#if defined(RAJA_ENABLE_TBB)

using TBBSortSorters =
  camp::list<
              PolicySort<RAJA::tbb_for_exec>,
              PolicySortPairs<RAJA::tbb_for_exec>
            >;

#endif

#if defined(RAJA_ENABLE_CUDA)

using CudaSortSorters =
  camp::list<
              PolicySort<RAJA::cuda_exec<128>>,
              PolicySortPairs<RAJA::cuda_exec<128>>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipSortSorters =
  camp::list<
              PolicySort<RAJA::hip_exec<128>>,
              PolicySortPairs<RAJA::hip_exec<128>>
            >;

#endif

#endif //__TEST_UNIT_ALGORITHM_SORT_HPP__

