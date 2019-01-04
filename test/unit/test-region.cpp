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
/// Source file containing tests for atomic operations
///

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"


template <typename RegionPolicy, typename loopPol>
void testRegion()
{

  int N = 100;
  int *A = new int[N];

  for (int i = 0; i < N; ++i) {
    A[i] = 0;
  }

  RAJA::region<RegionPolicy>([=]() {
    RAJA::forall<loopPol>(RAJA::RangeSegment(0, N), [=](int i) { A[i] += 1; });


    RAJA::forall<loopPol>(RAJA::RangeSegment(0, N), [=](int i) { A[i] += 1; });
  });

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(A[i], 2);
  }

  delete[] A;
}

template <typename ExecPol, typename LoopPol>
void testRegionPol()
{

  testRegion<ExecPol, LoopPol>();
}

TEST(Region, basic_Functions)
{

  testRegionPol<RAJA::seq_region, RAJA::loop_exec>();

#if defined(RAJA_ENABLE_OPENMP)
  testRegionPol<RAJA::omp_parallel_region, RAJA::omp_for_exec>();
#endif
}
