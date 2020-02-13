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

TEST(Atomic, basic_seq_Sort)
{
  testSortPol<RAJA::seq_exec>();
  // testSortPiarsPol<RAJA::seq_exec>();
}

TEST(Atomic, basic_loop_Sort)
{
  testSortPol<RAJA::loop_exec>();
  // testSortPairsPol<RAJA::loop_exec>();
}

#if defined(RAJA_ENABLE_OPENMP)

TEST(Atomic, basic_OpenMP_Sort)
{
  testSortPol<RAJA::omp_parallel_for_exec>();
  // testSortPairsPol<RAJA::omp_parallel_for_exec>();
}

#endif

#if defined(RAJA_ENABLE_TBB)

TEST(Atomic, basic_TBB_Sort)
{
  testSortPol<RAJA::tbb_for_exec>();
  // testSortPairsPol<RAJA::tbb_for_exec>();
}
#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Atomic, basic_CUDA_Sort)
{
  testSortPol<RAJA::cuda_exec<256>>();
  testSortPairsPol<RAJA::cuda_exec<256>>();
}

#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Atomic, basic_HIP_Sort)
{
  testSortPol<RAJA::hip_exec<256>>();
  testSortPairsPol<RAJA::hip_exec<256>>();
}

#endif

