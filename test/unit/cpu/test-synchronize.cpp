//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#if defined(RAJA_ENABLE_OPENMP)

TEST(SynchronizeTest, omp){

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
