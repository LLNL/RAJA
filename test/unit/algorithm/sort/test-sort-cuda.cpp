//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA sort with cuda policies
///

#include "../test-sort.hpp"

#if defined(RAJA_ENABLE_CUDA)

TEST(Sort, Sort_cuda)
{
  testSorter(PolicySort<RAJA::cuda_exec<128>>{"cuda"});
  testSorter(PolicySortPairs<RAJA::cuda_exec<128>>{"cuda"});
}

#endif

