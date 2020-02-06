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
}

// TODO implement omp sort
#if defined(RAJA_ENABLE_OPENMP) && 0

TEST(Atomic, basic_OpenMP_Sort)
{
  testSortPol<RAJA::omp_for_exec>();
}

#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Atomic, basic_CUDA_Sort)
{
  testSortPol<RAJA::cuda_exec<256>>();
}

#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Atomic, basic_HIP_Sort)
{
  testSortPol<RAJA::hip_exec<256>>();
}

#endif

