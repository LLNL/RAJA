//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#if defined(RAJA_ENABLE_OPENMP)

TEST(SynchronizeTest, omp)
{

  double test_val = 0.0;

#pragma omp parallel shared(test_val)
  {
    if (omp_get_thread_num() == 0) {
      test_val = 5.0;
    }

    RAJA::synchronize<RAJA::omp_synchronize>();

    EXPECT_EQ(test_val, 5.0);
  }
}

#endif
