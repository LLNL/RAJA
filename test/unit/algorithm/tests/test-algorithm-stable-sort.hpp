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
/// Header file containing Sorter classes for stable sort tests
///

#ifndef __TEST_UNIT_ALGORITHM_STABLE_SORT_HPP__
#define __TEST_UNIT_ALGORITHM_STABLE_SORT_HPP__

#include "test-algorithm-sort-utils.hpp"


template <typename policy>
struct PolicyStableSort : PolicySynchronize<policy>
{
  using sort_category     = stable_sort_tag;
  using sort_interface    = sort_interface_tag;
  using supports_resource = std::true_type;

  std::string m_name;

  PolicyStableSort() : m_name("RAJA::stable_sort<unknown>") {}

  PolicyStableSort(std::string const& policy_name)
      : m_name(std::string("RAJA::stable_sort<") + policy_name +
               std::string(">"))
  {}

  const char* name() { return m_name.c_str(); }

  template <typename... Args>
  void operator()(Args&&... args)
  {
    RAJA::stable_sort<policy>(std::forward<Args>(args)...);
  }
};

template <typename policy>
struct PolicyStableSortPairs : PolicySynchronize<policy>
{
  using sort_category     = stable_sort_tag;
  using sort_interface    = sort_pairs_interface_tag;
  using supports_resource = std::true_type;

  std::string m_name;

  PolicyStableSortPairs() : m_name("RAJA::stable_sort<unknown>[pairs]") {}

  PolicyStableSortPairs(std::string const& policy_name)
      : m_name(std::string("RAJA::stable_sort<") + policy_name +
               std::string(">[pairs]"))
  {}

  const char* name() { return m_name.c_str(); }

  template <typename... Args>
  void operator()(Args&&... args)
  {
    RAJA::stable_sort_pairs<policy>(std::forward<Args>(args)...);
  }
};

using SequentialStableSortSorters =
    camp::list<PolicyStableSort<RAJA::seq_exec>,
               PolicyStableSortPairs<RAJA::seq_exec>>;

#if defined(RAJA_ENABLE_OPENMP)

using OpenMPStableSortSorters =
    camp::list<PolicyStableSort<RAJA::omp_parallel_for_exec>,
               PolicyStableSortPairs<RAJA::omp_parallel_for_exec>>;

#endif

#if defined(RAJA_ENABLE_CUDA)

using CudaStableSortSorters =
    camp::list<PolicyStableSort<RAJA::cuda_exec<128>>,
               PolicyStableSortPairs<RAJA::cuda_exec<128>>,
               PolicyStableSort<RAJA::cuda_exec_explicit<128, 2>>>;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipStableSortSorters =
    camp::list<PolicyStableSort<RAJA::hip_exec<128>>,
               PolicyStableSortPairs<RAJA::hip_exec<128>>>;

#endif

#endif // __TEST_UNIT_ALGORITHM_STABLE_SORT_HPP__
